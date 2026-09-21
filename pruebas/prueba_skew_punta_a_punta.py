"""Training-serving skew de punta a punta (regla 3), pasando por Kafka, Spark, Redis y el servicio.

Compara lo que `/predecir` calcula **online** contra lo que el batch de Spark calculó **offline**
(`datos_features/historico_tarjeta.parquet`) para las mismas transacciones reales de `fraudTest`.

El servicio solo lee el estado *anterior* a la transacción que puntúa, y ese estado tiene que
haber llegado por el camino real (productor -> Kafka -> job de Spark Structured Streaming ->
Redis). Por eso la prueba trabaja por **olas**: en la ola k se puntúa la transacción k de cada
tarjeta (el estado incluye las k-1 anteriores), y recién después se la publica en Kafka y se
espera a que el job la aplique, antes de pasar a la ola k+1. La muestra es el *prefijo* de la
secuencia de cada tarjeta en `fraudTest`, así el estado previo (fraudTrain + las k-1 anteriores)
es exactamente el que vio el batch.

El servicio corre en proceso (FastAPI + tienda de Feast reales) con un modelo falso: acá se mide
el skew de las características, no el modelo. Necesita Kafka y Redis (`make tiempo-real-arriba`),
el dataset y los Parquet de `make calcular-historico` y `make arranque-en-frio`.
"""

import json
import math
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient
from confluent_kafka.cimpl import NewTopic
from fastapi.testclient import TestClient
from feast import FeatureStore

from fraude.caracteristicas.almacen_estado import leer_estados
from fraude.caracteristicas.arranque_en_frio import (
    RUTA_ESTADO_INICIAL,
    materializar_estado_inicial,
)
from fraude.caracteristicas.tienda import crear_tienda_online
from fraude.entrenamiento.carga import cargar_transacciones
from fraude.productor.mensajes import (
    clave_de_particion,
    iterar_filas,
    preparar_transacciones,
    serializar,
)
from fraude.servicio import principal
from fraude.servicio.esquemas import CaracteristicasHistoricasRespuesta, TransaccionEntrada
from fraude.servicio.modelo_actual import Campeon

CONEXION_REDIS = "localhost:6379,db=1"
RUTA_TRANSACCIONES = Path("datos/fraudTest.csv")
RUTA_HISTORICO = Path("datos_features/historico_tarjeta.parquet")

TARJETAS_CON_ESTADO = 25
TARJETAS_NUEVAS = 5
# Las tarjetas que no están en fraudTrain son de poca actividad (6 a 14 transacciones en fraudTest).
TRANSACCIONES_POR_TARJETA = 10
PLAZO_PRIMERA_OLA_S = 240  # incluye arrancar la JVM de Spark
PLAZO_OLA_S = 60

CAMPOS_PEDIDO = tuple(TransaccionEntrada.model_fields)
CARACTERISTICAS = tuple(CaracteristicasHistoricasRespuesta.model_fields)


class _ModeloFalso:
    def predict(self, model_input: pd.DataFrame) -> npt.NDArray[np.float64]:
        return np.zeros(len(model_input))


def _elegir_muestra(transacciones: pd.DataFrame, tarjetas_con_estado: set[int]) -> pd.DataFrame:
    """Las primeras K transacciones de las tarjetas que las tienen más concentradas en el tiempo.

    "Concentradas" para que las ventanas de 10 min, 1 h y 24 h tomen valores distintos de cero.
    Se incluyen tarjetas que no están en `fraudTrain`: arrancan sin estado y lo arma el streaming.
    """
    primeras = transacciones.groupby("numero_tarjeta").head(TRANSACCIONES_POR_TARJETA).copy()
    completas = primeras.groupby("numero_tarjeta").size() == TRANSACCIONES_POR_TARJETA
    primeras = primeras[primeras["numero_tarjeta"].isin(completas[completas].index)].copy()
    primeras["orden_en_tarjeta"] = primeras.groupby("numero_tarjeta").cumcount()
    fechas = primeras.groupby("numero_tarjeta")["fecha_hora_transaccion"]
    duracion = (fechas.max() - fechas.min()).sort_values()

    con_estado = [t for t in duracion.index if t in tarjetas_con_estado][:TARJETAS_CON_ESTADO]
    nuevas = [t for t in duracion.index if t not in tarjetas_con_estado][:TARJETAS_NUEVAS]
    return primeras[primeras["numero_tarjeta"].isin([*con_estado, *nuevas])]


def _iguales(online: object, offline: object) -> bool:
    """Mismo criterio que el resto de las pruebas: nulos iguales y flotantes con rel=1e-9."""
    nulo_online = online is None or (isinstance(online, float) and math.isnan(online))
    nulo_offline = offline is None or (isinstance(offline, float) and math.isnan(offline))
    if nulo_online or nulo_offline:
        return nulo_online and nulo_offline
    if isinstance(online, float) and isinstance(offline, float):
        return math.isclose(online, offline, rel_tol=1e-9)
    return bool(online == offline)


def _cola_del_log(ruta: Path, lineas: int = 30) -> str:
    return "\n".join(ruta.read_text(errors="replace").splitlines()[-lineas:])


def _esperar_a_que_se_apliquen(
    tienda: FeatureStore,
    esperados: dict[int, str],
    job: "subprocess.Popen[bytes]",
    log: Path,
    plazo_s: float,
) -> None:
    """Espera a que Redis tenga, para cada tarjeta, la transacción publicada como la última."""
    limite = time.monotonic() + plazo_s
    while True:
        estados = leer_estados(tienda, list(esperados))
        pendientes = [
            tarjeta
            for tarjeta, id_transaccion in esperados.items()
            if (estado := estados[tarjeta]) is None
            or estado.ultimo_id_transaccion != id_transaccion
        ]
        if not pendientes:
            return
        if job.poll() is not None:
            pytest.fail(
                f"El job de streaming terminó (código {job.returncode}):\n{_cola_del_log(log)}"
            )
        if time.monotonic() > limite:
            pytest.fail(
                f"{len(pendientes)} tarjetas sin aplicar tras {plazo_s} s:\n{_cola_del_log(log)}"
            )
        time.sleep(0.5)


def _detener(job: "subprocess.Popen[bytes]") -> None:
    """Mata al job y a la JVM de Spark que lanzó (están en su propio grupo de procesos)."""
    if job.poll() is not None:
        return
    os.killpg(job.pid, signal.SIGTERM)
    try:
        job.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(job.pid, signal.SIGKILL)
        job.wait()


@pytest.mark.integracion
def prueba_skew_de_punta_a_punta_por_kafka_spark_redis_y_servicio(
    tmp_path: Path, redis_de_prueba: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    for ruta in (RUTA_TRANSACCIONES, RUTA_HISTORICO, RUTA_ESTADO_INICIAL):
        if not ruta.exists():
            pytest.skip(f"Falta {ruta} (ver el docstring del módulo)")

    registro = tmp_path / "registro.db"
    tienda = crear_tienda_online(registro, CONEXION_REDIS)
    materializar_estado_inicial(tienda, RUTA_ESTADO_INICIAL)  # el arranque en frío real

    tarjetas_con_estado = set(
        pd.read_parquet(RUTA_ESTADO_INICIAL, columns=["numero_tarjeta"])["numero_tarjeta"]
    )
    transacciones = preparar_transacciones(cargar_transacciones(RUTA_TRANSACCIONES))
    muestra = _elegir_muestra(transacciones, tarjetas_con_estado)
    tarjetas_nuevas = set(muestra["numero_tarjeta"]) - tarjetas_con_estado
    assert len(tarjetas_nuevas) > 0, "la muestra tiene que incluir tarjetas sin estado inicial"
    offline = pd.read_parquet(RUTA_HISTORICO).set_index("id_transaccion")

    # El servicio real, con Redis y el registro de la prueba, y un modelo falso.
    monkeypatch.setenv("REDIS_CONEXION", CONEXION_REDIS)
    monkeypatch.setenv("REGISTRO_FEAST", str(registro))
    campeon = Campeon(modelo=_ModeloFalso(), umbral=0.5, version="prueba")  # type: ignore[arg-type]
    monkeypatch.setattr(principal, "cargar_campeon", lambda: campeon)

    topico = f"prueba_skew_{uuid.uuid4().hex[:8]}"
    admin = AdminClient({"bootstrap.servers": "localhost:9092"})
    admin.create_topics([NewTopic(topico, num_partitions=3, replication_factor=1)])[topico].result()
    log = tmp_path / "job.log"
    productor = Producer({"bootstrap.servers": "localhost:9092"})
    resultados: list[dict[str, Any]] = []

    with log.open("wb") as salida_del_job:
        job = subprocess.Popen(
            [
                sys.executable, "-m", "fraude.tiempo_real.streaming_estado",
                "--topico", topico,
                "--checkpoint", str(tmp_path / "checkpoint"),
                "--registro", str(registro),
                "--conexion-redis", CONEXION_REDIS,
                "--intervalo", "1",
            ],
            stdout=salida_del_job,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )  # fmt: skip
        try:
            with TestClient(principal.app) as cliente:
                olas = list(muestra.groupby("orden_en_tarjeta"))
                for numero_ola, (_, ola) in enumerate(olas):
                    filas = list(iterar_filas(ola))

                    # 1) Puntuar con el estado previo (el que llegó por Kafka -> Spark -> Redis).
                    for fila in filas:
                        cuerpo = json.loads(serializar(fila))
                        respuesta = cliente.post(
                            "/predecir", json={campo: cuerpo[campo] for campo in CAMPOS_PEDIDO}
                        )
                        assert respuesta.status_code == 200, respuesta.text
                        datos = respuesta.json()
                        resultados.append(
                            {
                                "id_transaccion": fila["id_transaccion"],
                                "numero_tarjeta": fila["numero_tarjeta"],
                                "orden_en_tarjeta": numero_ola,
                                "tarjeta_con_historia": datos["tarjeta_con_historia"],
                                **datos["caracteristicas_historicas"],
                            }
                        )

                    # 2) Publicarlas y 3) esperar a que el job las aplique (ola siguiente).
                    if numero_ola == len(olas) - 1:
                        break
                    for fila in filas:
                        productor.produce(
                            topico, key=clave_de_particion(fila), value=serializar(fila)
                        )
                    assert productor.flush(30) == 0
                    _esperar_a_que_se_apliquen(
                        tienda,
                        {f["numero_tarjeta"]: f["id_transaccion"] for f in filas},
                        job,
                        log,
                        PLAZO_PRIMERA_OLA_S if numero_ola == 0 else PLAZO_OLA_S,
                    )
        finally:
            _detener(job)
            admin.delete_topics([topico])

    obtenido = pd.DataFrame(resultados)
    assert len(obtenido) == len(muestra)

    # El estado de una tarjeta nueva lo construyó el streaming: sin historia en la primera
    # transacción y con historia desde la segunda.
    nuevas = obtenido[obtenido["numero_tarjeta"].isin(tarjetas_nuevas)]
    assert not nuevas[nuevas["orden_en_tarjeta"] == 0]["tarjeta_con_historia"].any()
    assert nuevas[nuevas["orden_en_tarjeta"] > 0]["tarjeta_con_historia"].all()

    # Sin variación no hay prueba: las ventanas tienen que haber tomado valores distintos.
    assert obtenido["cantidad_transacciones_24h"].nunique() > 2

    esperado = offline.loc[obtenido["id_transaccion"], list(CARACTERISTICAS)]
    diferencias = [
        (fila["id_transaccion"], columna, fila[columna], esperado.iloc[i][columna])
        for i, fila in enumerate(obtenido.to_dict("records"))
        for columna in CARACTERISTICAS
        if not _iguales(fila[columna], esperado.iloc[i][columna])
    ]
    assert not diferencias, (
        f"{len(diferencias)} celdas distintas de {len(obtenido) * len(CARACTERISTICAS)}; "
        f"primeras: {diferencias[:10]}"
    )
