"""Modelo de gradient boosting (LightGBM) sobre las características básicas.

El desbalance de clases se maneja con pesos por clase (`class_weight="balanced"`),
nunca remuestreando: remuestrear cambiaría la distribución de montos y probabilidades
que necesita la métrica de costo para ser realista.
"""

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd

COLUMNAS_NUMERICAS = [
    "monto",
    "hora_del_dia",
    "dia_de_la_semana",
    "edad_titular",
    "distancia_cliente_comercio_km",
    "poblacion_ciudad",
]
COLUMNAS_CATEGORICAS = ["categoria", "genero"]
COLUMNAS_CARACTERISTICAS = [*COLUMNAS_NUMERICAS, *COLUMNAS_CATEGORICAS]


def preparar_matriz(transacciones: pd.DataFrame) -> pd.DataFrame:
    """Selecciona las columnas de características y tipa las categóricas para LightGBM."""
    matriz = transacciones[COLUMNAS_CARACTERISTICAS].copy()
    for columna in COLUMNAS_CATEGORICAS:
        matriz[columna] = matriz[columna].astype("category")
    return matriz


def entrenar(transacciones_entrenamiento: pd.DataFrame, semilla: int = 0) -> lgb.LGBMClassifier:
    """Entrena un LightGBM con pesos por clase sobre las transacciones de entrenamiento."""
    x = preparar_matriz(transacciones_entrenamiento)
    y = transacciones_entrenamiento["es_fraude"].to_numpy()
    modelo = lgb.LGBMClassifier(
        class_weight="balanced",
        random_state=semilla,
        n_estimators=200,
        verbosity=-1,
    )
    modelo.fit(x, y, categorical_feature=COLUMNAS_CATEGORICAS)
    return modelo


def predecir_probabilidad(
    modelo: lgb.LGBMClassifier, transacciones: pd.DataFrame
) -> npt.NDArray[np.float64]:
    """Probabilidad de fraude que asigna el modelo a cada transacción."""
    matriz = preparar_matriz(transacciones)
    resultado = modelo.predict_proba(matriz)
    # predict_proba() puede devolver una matriz dispersa segun el tipo de entrada;
    # con un DataFrame denso siempre devuelve un ndarray.
    assert isinstance(resultado, np.ndarray)
    probabilidades: npt.NDArray[np.float64] = resultado[:, 1]
    return probabilidades
