"""Distancias sobre la superficie terrestre, en Python puro (sin pandas ni Spark).

La usa el servicio en el momento de la solicitud. Las versiones vectorizadas del entrenamiento
(`entrenamiento.caracteristicas_basicas`, pandas) y del batch (`lotes.caracteristicas_historicas`,
Spark) implementan la misma fórmula sobre el mismo radio; la prueba de skew las compara.
"""

from math import asin, cos, radians, sin, sqrt

RADIO_TIERRA_KM = 6371.0


def distancia_haversine_km(
    latitud_1: float, longitud_1: float, latitud_2: float, longitud_2: float
) -> float:
    """Distancia en línea recta (círculo máximo) entre dos coordenadas, en kilómetros."""
    fi_1, fi_2 = radians(latitud_1), radians(latitud_2)
    delta_fi = radians(latitud_2 - latitud_1)
    delta_lambda = radians(longitud_2 - longitud_1)
    a = sin(delta_fi / 2) ** 2 + cos(fi_1) * cos(fi_2) * sin(delta_lambda / 2) ** 2
    return 2 * RADIO_TIERRA_KM * asin(sqrt(a))
