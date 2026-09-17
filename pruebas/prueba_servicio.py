from collections.abc import Iterator
from datetime import datetime

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraude.servicio import principal
from fraude.servicio.modelo_actual import Campeon, cargar_campeon


class _ModeloFalso:
    """Reemplaza al PyFuncModel real: devuelve siempre la misma probabilidad."""

    def predict(self, model_input: pd.DataFrame) -> npt.NDArray[np.float64]:
        return np.full(len(model_input), 0.85)


def _transaccion_de_ejemplo() -> dict[str, object]:
    return {
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


@pytest.fixture
def cliente(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    campeon_falso = Campeon(modelo=_ModeloFalso(), umbral=0.5, version="7")  # type: ignore[arg-type]
    monkeypatch.setattr(principal, "cargar_campeon", lambda: campeon_falso)
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
