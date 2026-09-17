"""Características point-in-time calculables por fila, sin historia de la tarjeta.

Para la línea base de la semana 1 alcanza con lo que ya viene en cada transacción:
monto, categoría, momento del día y distancia entre el cliente y el comercio. Las
características que necesitan agregados históricos (conteos en ventanas de tiempo,
promedios por tarjeta) se calculan más adelante con PySpark/Spark Streaming y se
sirven desde Feast — no están acá para no duplicar esa lógica de antemano.
"""

import numpy as np
import pandas as pd

RADIO_TIERRA_KM = 6371.0


def _distancia_haversine_km(
    lat_1: pd.Series, long_1: pd.Series, lat_2: pd.Series, long_2: pd.Series
) -> pd.Series:
    """Distancia en línea recta entre dos coordenadas, en kilómetros."""
    fi_1, fi_2 = np.radians(lat_1), np.radians(lat_2)
    delta_fi = np.radians(lat_2 - lat_1)
    delta_lambda = np.radians(long_2 - long_1)
    a = np.sin(delta_fi / 2) ** 2 + np.cos(fi_1) * np.cos(fi_2) * np.sin(delta_lambda / 2) ** 2
    return pd.Series(2 * RADIO_TIERRA_KM * np.arcsin(np.sqrt(a)), index=lat_1.index)


def agregar_caracteristicas_basicas(transacciones: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia de `transacciones` con las características básicas agregadas."""
    resultado = transacciones.copy()
    resultado["hora_del_dia"] = resultado["fecha_hora_transaccion"].dt.hour
    resultado["dia_de_la_semana"] = resultado["fecha_hora_transaccion"].dt.dayofweek
    resultado["edad_titular"] = (
        resultado["fecha_hora_transaccion"] - resultado["fecha_nacimiento"]
    ).dt.days / 365.25
    resultado["distancia_cliente_comercio_km"] = _distancia_haversine_km(
        resultado["latitud_cliente"],
        resultado["longitud_cliente"],
        resultado["latitud_comercio"],
        resultado["longitud_comercio"],
    )
    return resultado
