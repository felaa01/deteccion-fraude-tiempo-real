from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from feast import FeatureStore
from redis.exceptions import ConnectionError as ErrorConexionRedis

from fraude.caracteristicas.estado_tarjeta import EstadoTarjeta, a_marca_unix
from fraude.servicio import principal
from fraude.servicio.modelo_actual import Campeon, cargar_campeon

TARJETA = 4_000_111_122_223_333


class _ModeloFalso:
    """Reemplaza al PyFuncModel real: devuelve siempre la misma probabilidad."""

    def __init__(self) -> None:
        self.columnas_recibidas: list[str] = []

    def predict(self, model_input: pd.DataFrame) -> npt.NDArray[np.float64]:
        self.columnas_recibidas = list(model_input.columns)
        return np.full(len(model_input), 0.85)


def _transaccion_de_ejemplo() -> dict[str, object]:
    return {
        "numero_tarjeta": TARJETA,
        "fecha_hora_transaccion": datetime(2024, 6, 1, 13, 30, 0).isoformat(),
        "monto": 150.0,
        "categoria": "shopping_net",
        "genero": "F",
        "fecha_nacimiento": datetime(1990, 1, 1).isoformat(),
        "latitud_cliente": -34.6,
        "longitud_cliente": -58.4,
        "latitud_comercio": -34.5,
        "longitud_comercio": -58.5,
        "poblacion_ciudad": 500_000,
    }


def _estado_con_una_transaccion_a_las(hora: datetime) -> EstadoTarjeta:
    """Estado de una tarjeta con una única transacción previa de 50 en `shopping_net`."""
    marca = a_marca_unix(hora)
    return EstadoTarjeta(
        cantidad_total=1,
        monto_total=50.0,
        categorias_vistas=("shopping_net",),
        marcas_recientes=(marca,),
        montos_recientes=(50.0,),
        ultima_marca=marca,
        ultimo_id_transaccion="tx-1",
        ultima_latitud_comercio=-34.6,
        ultima_longitud_comercio=-58.4,
    )


@pytest.fixture
def modelo_falso() -> _ModeloFalso:
    return _ModeloFalso()


@pytest.fixture
def estados_en_redis() -> dict[int, EstadoTarjeta]:
    """El "contenido" de Redis: los tests agregan tarjetas acá."""
    return {}


@pytest.fixture
def cliente(
    monkeypatch: pytest.MonkeyPatch,
    modelo_falso: _ModeloFalso,
    estados_en_redis: dict[int, EstadoTarjeta],
) -> Iterator[TestClient]:
    campeon_falso = Campeon(modelo=modelo_falso, umbral=0.5, version="7")  # type: ignore[arg-type]

    def leer_estados_falso(
        tienda: FeatureStore, tarjetas: Sequence[int]
    ) -> Mapping[int, EstadoTarjeta | None]:
        return {tarjeta: estados_en_redis.get(tarjeta) for tarjeta in tarjetas}

    monkeypatch.setattr(principal, "cargar_campeon", lambda: campeon_falso)
    monkeypatch.setattr(principal, "cargar_tienda", lambda: object())
    monkeypatch.setattr(principal, "leer_estados", leer_estados_falso)
    with TestClient(principal.app) as client:
        yield client


def prueba_predecir_devuelve_fraude_cuando_supera_el_umbral(cliente: TestClient) -> None:
    respuesta = cliente.post("/predecir", json=_transaccion_de_ejemplo())

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["probabilidad_fraude"] == pytest.approx(0.85)
    assert cuerpo["es_fraude"] is True
    assert cuerpo["umbral_usado"] == 0.5
    assert cuerpo["version_modelo"] == "7"


def prueba_predecir_calcula_las_caracteristicas_historicas_con_el_estado_de_redis(
    cliente: TestClient, estados_en_redis: dict[int, EstadoTarjeta]
) -> None:
    # Cinco minutos antes de la transacción de ejemplo (13:30).
    estados_en_redis[TARJETA] = _estado_con_una_transaccion_a_las(datetime(2024, 6, 1, 13, 25))

    respuesta = cliente.post("/predecir", json=_transaccion_de_ejemplo())

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["tarjeta_con_historia"] is True
    historicas = cuerpo["caracteristicas_historicas"]
    assert historicas["cantidad_transacciones_10min"] == 1
    assert historicas["cantidad_transacciones_24h"] == 1
    assert historicas["monto_acumulado_tarjeta"] == pytest.approx(50.0)
    assert historicas["ratio_monto_promedio_tarjeta"] == pytest.approx(3.0)  # 150 / 50
    assert historicas["categoria_nueva_para_tarjeta"] is False
    assert historicas["distancia_transaccion_anterior_km"] > 0
    assert historicas["velocidad_implicita_kmh"] > 0


def prueba_predecir_trata_como_nueva_a_una_tarjeta_sin_estado(cliente: TestClient) -> None:
    respuesta = cliente.post("/predecir", json=_transaccion_de_ejemplo())

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["tarjeta_con_historia"] is False
    historicas = cuerpo["caracteristicas_historicas"]
    assert historicas["cantidad_transacciones_24h"] == 0
    assert historicas["monto_acumulado_tarjeta"] is None
    assert historicas["categoria_nueva_para_tarjeta"] is True


def prueba_predecir_devuelve_409_si_la_solicitud_es_anterior_al_estado(
    cliente: TestClient, estados_en_redis: dict[int, EstadoTarjeta]
) -> None:
    # El estado ya incorporó una transacción posterior a la que se quiere puntuar.
    estados_en_redis[TARJETA] = _estado_con_una_transaccion_a_las(datetime(2024, 6, 1, 14, 0))

    respuesta = cliente.post("/predecir", json=_transaccion_de_ejemplo())

    assert respuesta.status_code == 409


def prueba_predecir_devuelve_503_si_redis_no_responde(
    cliente: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def redis_caido(tienda: FeatureStore, tarjetas: Sequence[int]) -> None:
        raise ErrorConexionRedis("Connection refused")

    monkeypatch.setattr(principal, "leer_estados", redis_caido)

    respuesta = cliente.post("/predecir", json=_transaccion_de_ejemplo())

    assert respuesta.status_code == 503


def prueba_predecir_no_le_pasa_el_numero_de_tarjeta_al_modelo(
    cliente: TestClient, modelo_falso: _ModeloFalso
) -> None:
    cliente.post("/predecir", json=_transaccion_de_ejemplo())

    assert "numero_tarjeta" not in modelo_falso.columnas_recibidas


def prueba_predecir_rechaza_un_monto_invalido(cliente: TestClient) -> None:
    transaccion = _transaccion_de_ejemplo()
    transaccion["monto"] = -10.0

    respuesta = cliente.post("/predecir", json=transaccion)

    assert respuesta.status_code == 422


def prueba_salud_responde_ok(cliente: TestClient) -> None:
    respuesta = cliente.get("/salud")

    assert respuesta.status_code == 200
    assert respuesta.json() == {"estado": "ok"}


@pytest.mark.integracion
def prueba_carga_el_campeon_vigente_desde_mlflow() -> None:
    campeon = cargar_campeon()

    assert 0.0 <= campeon.umbral <= 1.0
    assert campeon.version
