import json
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient
from confluent_kafka.cimpl import NewTopic
from feast import FeatureStore, Project
from pyspark.sql import SparkSession

from fraude.caracteristicas.almacen_estado import leer_estados
from fraude.caracteristicas.definiciones import (
    TARJETA,
    crear_fuente_estado_inicial,
    crear_vista_estado_tarjeta,
)
from fraude.caracteristicas.estado_tarjeta import EstadoTarjeta, TransaccionTarjeta
from fraude.caracteristicas.tienda import crear_tienda_online
from fraude.tiempo_real.lote_estado import (
    TransaccionEntrante,
    aplicar_transacciones,
    procesar_lote,
)
from fraude.tiempo_real.streaming_estado import a_transacciones_entrantes, parsear_mensajes

CONEXION_REDIS_DE_PRUEBA = "localhost:6379,db=1"
PROYECTO_DE_PRUEBA = "prueba_streaming_estado"


def _entrante(
    tarjeta: int, id_transaccion: str, marca: int, monto: float = 10.0, categoria: str = "comida"
) -> TransaccionEntrante:
    return TransaccionEntrante(tarjeta, TransaccionTarjeta(id_transaccion, marca, monto, categoria))


def prueba_aplicar_transacciones_ordena_por_marca_dentro_de_cada_tarjeta() -> None:
    desordenadas = [_entrante(1, "b", 200), _entrante(1, "a", 100), _entrante(2, "c", 50)]

    estados = aplicar_transacciones({}, desordenadas)

    assert estados[1].ultimo_id_transaccion == "b"
    assert estados[1].marcas_recientes == (100, 200)
    assert estados[2].cantidad_total == 1


def prueba_un_lote_ya_aplicado_no_cambia_nada() -> None:
    """Reintento de un lote tras una falla (al menos una vez): no se cuenta dos veces."""
    lote = [_entrante(1, "a", 100), _entrante(1, "b", 200)]
    estados = aplicar_transacciones({}, lote)

    repetido = aplicar_transacciones(estados, lote)

    assert repetido == {}


def prueba_un_lote_parcialmente_repetido_solo_aplica_lo_nuevo() -> None:
    estados = aplicar_transacciones({}, [_entrante(1, "a", 100)])

    nuevos = aplicar_transacciones(estados, [_entrante(1, "a", 100), _entrante(1, "b", 200)])

    assert nuevos[1].cantidad_total == 2


def prueba_tarjeta_sin_estado_previo_arranca_de_cero() -> None:
    estados = aplicar_transacciones({7: None}, [_entrante(7, "a", 100, monto=5.0)])

    assert estados[7].cantidad_total == 1
    assert estados[7].monto_total == 5.0


def _mensajes(sesion_spark: SparkSession, valores: list[bytes]) -> pd.DataFrame:
    return parsear_mensajes(
        sesion_spark.createDataFrame([(v,) for v in valores], ["value"])
    ).toPandas()


def prueba_parsear_mensajes_convierte_json_a_columnas_tipadas(sesion_spark: SparkSession) -> None:
    valor = (
        b'{"id_transaccion":"tx1","numero_tarjeta":123,"fecha_hora_transaccion":'
        b'"2020-06-21T12:14:25","monto":29.84,"categoria":"ocio","otro_campo":"se ignora"}'
    )

    resultado = _mensajes(sesion_spark, [valor])

    fila = resultado.iloc[0]
    assert (fila["numero_tarjeta"], fila["id_transaccion"], fila["monto"]) == (123, "tx1", 29.84)
    assert fila["marca_tiempo"] == int(datetime(2020, 6, 21, 12, 14, 25, tzinfo=UTC).timestamp())


def prueba_parsear_mensajes_no_depende_de_la_zona_horaria_de_la_sesion(
    sesion_spark: SparkSession,
) -> None:
    """Regresión del bug de la semana 3: la fecha es UTC, no la zona de la sesión."""
    valor = (
        b'{"id_transaccion":"tx1","numero_tarjeta":1,"fecha_hora_transaccion":'
        b'"2020-06-21T12:14:25","monto":1.0,"categoria":"ocio"}'
    )
    zona_original = sesion_spark.conf.get("spark.sql.session.timeZone")
    sesion_spark.conf.set("spark.sql.session.timeZone", "America/Argentina/Buenos_Aires")
    try:
        resultado = _mensajes(sesion_spark, [valor])
    finally:
        if zona_original is not None:
            sesion_spark.conf.set("spark.sql.session.timeZone", zona_original)

    assert resultado.iloc[0]["marca_tiempo"] == 1_592_741_665


def prueba_mensajes_invalidos_se_descartan_y_se_cuentan(sesion_spark: SparkSession) -> None:
    valido = (
        b'{"id_transaccion":"tx1","numero_tarjeta":1,"fecha_hora_transaccion":'
        b'"2020-06-21T12:14:25","monto":1.0,"categoria":"ocio"}'
    )
    sin_monto = (
        b'{"id_transaccion":"tx2","numero_tarjeta":1,'
        b'"fecha_hora_transaccion":"2020-06-21T12:14:25"}'
    )
    resultado = _mensajes(sesion_spark, [valido, b"esto no es json", sin_monto]).astype(object)
    # `toPandas` deja los nulos como NaN; el job real trabaja con `None` (filas de Spark).
    filas = [
        tuple(None if pd.isna(valor) else valor for valor in fila)
        for fila in resultado.itertuples(index=False)
    ]

    entrantes, invalidas = a_transacciones_entrantes(filas)

    assert [e.transaccion.id_transaccion for e in entrantes] == ["tx1"]
    assert invalidas == 2


def _tienda_de_prueba(directorio: Path) -> FeatureStore:
    tienda = crear_tienda_online(
        directorio / "registro.db", CONEXION_REDIS_DE_PRUEBA, proyecto=PROYECTO_DE_PRUEBA
    )
    vista = crear_vista_estado_tarjeta(crear_fuente_estado_inicial(directorio / "inicial.parquet"))
    tienda.apply([Project(name=PROYECTO_DE_PRUEBA), TARJETA, vista])
    return tienda


@pytest.mark.integracion
def prueba_procesar_lote_lee_actualiza_y_escribe_en_redis(
    tmp_path: Path, redis_de_prueba: None
) -> None:
    tienda = _tienda_de_prueba(tmp_path)

    primero = procesar_lote(tienda, [_entrante(1, "a", 100), _entrante(2, "b", 100)])
    segundo = procesar_lote(tienda, [_entrante(1, "c", 200)])
    repetido = procesar_lote(tienda, [_entrante(1, "c", 200)])

    assert (primero.tarjetas, primero.tarjetas_actualizadas) == (2, 2)
    assert (segundo.tarjetas, segundo.tarjetas_actualizadas) == (1, 1)
    assert repetido.tarjetas_actualizadas == 0
    leidos = leer_estados(tienda, [1, 2])
    estado_1, estado_2 = leidos[1], leidos[2]
    assert estado_1 is not None
    assert estado_2 is not None
    assert (estado_1.cantidad_total, estado_2.cantidad_total) == (2, 1)


def _publicar(topico: str, transacciones: pd.DataFrame) -> None:
    productor = Producer({"bootstrap.servers": "localhost:9092"})
    for fila in transacciones.to_dict("records"):
        fecha = datetime.fromtimestamp(fila["marca_tiempo_unix"], tz=UTC)
        cuerpo = json.dumps(
            {
                "id_transaccion": fila["id_transaccion"],
                "numero_tarjeta": fila["numero_tarjeta"],
                "fecha_hora_transaccion": fecha.strftime("%Y-%m-%dT%H:%M:%S"),
                "monto": fila["monto"],
                "categoria": fila["categoria"],
            }
        )
        productor.produce(topico, key=str(fila["numero_tarjeta"]), value=cuerpo.encode())
    productor.produce(topico, key="basura", value=b"esto no es json")
    assert productor.flush(30) == 0


def _correr_job(topico: str, checkpoint: Path, registro: Path) -> None:
    subprocess.run(
        [
            sys.executable, "-m", "fraude.tiempo_real.streaming_estado",
            "--topico", topico,
            "--checkpoint", str(checkpoint),
            "--registro", str(registro),
            "--conexion-redis", CONEXION_REDIS_DE_PRUEBA,
            "--proyecto", PROYECTO_DE_PRUEBA,
            "--hasta-agotar",
        ],
        check=True,
        timeout=300,
    )  # fmt: skip


@pytest.mark.integracion
def prueba_streaming_de_punta_a_punta(
    tmp_path: Path, redis_de_prueba: None, transacciones_aleatorias: pd.DataFrame
) -> None:
    """Kafka -> Spark Structured Streaming -> Redis: el estado final es el secuencial.

    Corre el job dos veces con checkpoints distintos: la segunda relee todo el tópico desde el
    principio (como tras perder el checkpoint) y no debe cambiar el estado, porque
    `actualizar_estado` ignora lo ya aplicado.
    """
    tienda = _tienda_de_prueba(tmp_path)
    topico = f"prueba_{uuid.uuid4().hex[:8]}"
    admin = AdminClient({"bootstrap.servers": "localhost:9092"})
    admin.create_topics([NewTopic(topico, num_partitions=3, replication_factor=1)])[topico].result()
    try:
        _publicar(topico, transacciones_aleatorias)
        esperado: dict[int, EstadoTarjeta] = aplicar_transacciones(
            {},
            [
                _entrante(
                    f["numero_tarjeta"],
                    f["id_transaccion"],
                    f["marca_tiempo_unix"],
                    f["monto"],
                    f["categoria"],
                )
                for f in transacciones_aleatorias.to_dict("records")
            ],
        )

        _correr_job(topico, tmp_path / "checkpoint_1", tmp_path / "registro.db")
        despues_de_la_primera = leer_estados(tienda, list(esperado))
        _correr_job(topico, tmp_path / "checkpoint_2", tmp_path / "registro.db")
        despues_de_la_segunda = leer_estados(tienda, list(esperado))
    finally:
        admin.delete_topics([topico])

    assert despues_de_la_primera == esperado
    assert despues_de_la_segunda == esperado
