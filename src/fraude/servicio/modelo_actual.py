"""Carga el campeón vigente (modelo pyfunc + umbral) desde el registro de MLflow.

Se carga una sola vez, al arrancar el servicio (ver `fraude.servicio.principal`). Si mientras
tanto se promueve un nuevo campeón en el registro, el servicio lo toma recién en el próximo
reinicio — el refresco en caliente es un problema de despliegue (semana 6, Kubernetes), no de
esta API.
"""

import os
from dataclasses import dataclass

import mlflow
import mlflow.pyfunc
from dotenv import load_dotenv
from mlflow.pyfunc import PyFuncModel

from fraude.entrenamiento.entrenar import ALIAS_CAMPEON, NOMBRE_MODELO_REGISTRADO


@dataclass(frozen=True)
class Campeon:
    """El modelo campeón cargado, listo para predecir, junto con su umbral de decisión."""

    modelo: PyFuncModel
    umbral: float
    version: str


def cargar_campeon() -> Campeon:
    """Trae la versión con el alias `campeon` y el umbral que se eligió al entrenarla."""
    load_dotenv()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    cliente = mlflow.MlflowClient()
    version = cliente.get_model_version_by_alias(NOMBRE_MODELO_REGISTRADO, ALIAS_CAMPEON)
    assert version.run_id is not None
    umbral = float(cliente.get_run(version.run_id).data.params["umbral_probabilidad"])
    modelo = mlflow.pyfunc.load_model(f"models:/{NOMBRE_MODELO_REGISTRADO}@{ALIAS_CAMPEON}")

    return Campeon(modelo=modelo, umbral=umbral, version=version.version)
