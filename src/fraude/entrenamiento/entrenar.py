"""Orquesta el entrenamiento de la semana 1: línea base, LightGBM y registro en MLflow.

Se corre manualmente por ahora (`uv run python -m fraude.entrenamiento.entrenar`). Más
adelante esto lo dispara el DAG semanal de Airflow, con las etiquetas disponibles hasta
ese momento.
"""

import os
from pathlib import Path

import mlflow
import mlflow.lightgbm
import numpy as np
import numpy.typing as npt
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import average_precision_score

import fraude
from fraude.entrenamiento.caracteristicas_basicas import agregar_caracteristicas_basicas
from fraude.entrenamiento.carga import cargar_transacciones
from fraude.entrenamiento.costo import costo_total, elegir_umbral
from fraude.entrenamiento.division import dividir_temporalmente
from fraude.entrenamiento.linea_base import elegir_umbral_de_monto
from fraude.entrenamiento.modelo import entrenar, predecir_probabilidad

COSTO_REVISION = 10.0
NOMBRE_EXPERIMENTO = "deteccion-fraude"
NOMBRE_MODELO_REGISTRADO = "deteccion-fraude-lightgbm"
ALIAS_CAMPEON = "campeon"


def _raiz_repo() -> Path:
    return Path(fraude.__file__).resolve().parents[2]


def _correr_linea_base(
    es_fraude_validacion: npt.NDArray[np.bool_],
    monto_validacion: npt.NDArray[np.float64],
    es_fraude_prueba: npt.NDArray[np.bool_],
    monto_prueba: npt.NDArray[np.float64],
) -> None:
    with mlflow.start_run(run_name="linea-base-monto"):
        umbral = elegir_umbral_de_monto(es_fraude_validacion, monto_validacion, COSTO_REVISION)
        prediccion_prueba = monto_prueba >= umbral.umbral
        costo_prueba = costo_total(
            es_fraude_prueba, prediccion_prueba, monto_prueba, COSTO_REVISION
        )

        mlflow.log_param("umbral_monto", umbral.umbral)
        mlflow.log_metric("costo_validacion", umbral.costo)
        mlflow.log_metric("costo_prueba", costo_prueba)
        print(f"Línea base -> umbral monto={umbral.umbral:.2f} costo_prueba={costo_prueba:.2f}")


def _correr_lightgbm(
    entrenamiento: pd.DataFrame,
    validacion: pd.DataFrame,
    prueba: pd.DataFrame,
    es_fraude_validacion: npt.NDArray[np.bool_],
    monto_validacion: npt.NDArray[np.float64],
    es_fraude_prueba: npt.NDArray[np.bool_],
    monto_prueba: npt.NDArray[np.float64],
) -> None:
    with mlflow.start_run(run_name="lightgbm"):
        modelo = entrenar(entrenamiento)
        probabilidad_validacion = predecir_probabilidad(modelo, validacion)
        umbral = elegir_umbral(
            es_fraude_validacion,
            probabilidad_validacion,
            monto_validacion,
            costo_revision=COSTO_REVISION,
        )

        probabilidad_prueba = predecir_probabilidad(modelo, prueba)
        prediccion_prueba = probabilidad_prueba >= umbral.umbral
        costo_prueba = costo_total(
            es_fraude_prueba, prediccion_prueba, monto_prueba, COSTO_REVISION
        )
        pr_auc = average_precision_score(es_fraude_prueba, probabilidad_prueba)

        mlflow.log_param("umbral_probabilidad", umbral.umbral)
        mlflow.log_metric("costo_validacion", umbral.costo)
        mlflow.log_metric("costo_prueba", costo_prueba)
        mlflow.log_metric("pr_auc_prueba", float(pr_auc))
        info_modelo = mlflow.lightgbm.log_model(
            modelo, name="modelo", registered_model_name=NOMBRE_MODELO_REGISTRADO
        )
        print(
            f"LightGBM -> umbral={umbral.umbral:.3f} "
            f"costo_prueba={costo_prueba:.2f} pr_auc={pr_auc:.4f}"
        )

        if info_modelo.registered_model_version is not None:
            cliente = mlflow.MlflowClient()
            cliente.set_registered_model_alias(
                NOMBRE_MODELO_REGISTRADO, ALIAS_CAMPEON, info_modelo.registered_model_version
            )


def main() -> None:
    load_dotenv(_raiz_repo() / ".env")
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(NOMBRE_EXPERIMENTO)

    ruta_datos = _raiz_repo() / "datos"
    crudo_entrenamiento = cargar_transacciones(ruta_datos / "fraudTrain.csv")
    crudo_prueba = cargar_transacciones(ruta_datos / "fraudTest.csv")
    conjuntos = dividir_temporalmente(crudo_entrenamiento, crudo_prueba)

    entrenamiento = agregar_caracteristicas_basicas(conjuntos.entrenamiento)
    validacion = agregar_caracteristicas_basicas(conjuntos.validacion)
    prueba = agregar_caracteristicas_basicas(conjuntos.prueba)

    es_fraude_validacion: npt.NDArray[np.bool_] = validacion["es_fraude"].astype(bool).to_numpy()
    monto_validacion: npt.NDArray[np.float64] = validacion["monto"].to_numpy()
    es_fraude_prueba: npt.NDArray[np.bool_] = prueba["es_fraude"].astype(bool).to_numpy()
    monto_prueba: npt.NDArray[np.float64] = prueba["monto"].to_numpy()

    _correr_linea_base(es_fraude_validacion, monto_validacion, es_fraude_prueba, monto_prueba)
    _correr_lightgbm(
        entrenamiento,
        validacion,
        prueba,
        es_fraude_validacion,
        monto_validacion,
        es_fraude_prueba,
        monto_prueba,
    )


if __name__ == "__main__":
    main()
