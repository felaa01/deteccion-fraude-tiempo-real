"""Lectura y escritura del estado por tarjeta en el almacén online de Feast (Redis).

Traduce entre `EstadoTarjeta` (el dataclass puro de `estado_tarjeta`) y el DataFrame y el
diccionario que usa la API de Feast, para que ni el job de streaming ni el servicio
conozcan esos detalles.
"""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from feast import FeatureStore
from feast.data_source import PushMode

from fraude.caracteristicas.definiciones import (
    COLUMNA_TIEMPO_ESTADO,
    NOMBRE_FUENTE_PUSH_ESTADO,
    NOMBRE_VISTA_ESTADO_TARJETA,
)
from fraude.caracteristicas.estado_tarjeta import EstadoTarjeta

CAMPOS_ESTADO = (
    "cantidad_total",
    "monto_total",
    "categorias_vistas",
    "marcas_recientes",
    "montos_recientes",
    "ultima_marca",
    "ultimo_id_transaccion",
    "ultima_latitud_comercio",
    "ultima_longitud_comercio",
)


def estados_a_dataframe(estados: Mapping[int, EstadoTarjeta]) -> pd.DataFrame:
    """Una fila por tarjeta, con la columna de tiempo que Feast usa como `timestamp_field`."""
    filas = [
        {
            "numero_tarjeta": tarjeta,
            "cantidad_total": estado.cantidad_total,
            "monto_total": estado.monto_total,
            "categorias_vistas": list(estado.categorias_vistas),
            "marcas_recientes": list(estado.marcas_recientes),
            "montos_recientes": list(estado.montos_recientes),
            "ultima_marca": estado.ultima_marca,
            "ultimo_id_transaccion": estado.ultimo_id_transaccion,
            "ultima_latitud_comercio": estado.ultima_latitud_comercio,
            "ultima_longitud_comercio": estado.ultima_longitud_comercio,
            COLUMNA_TIEMPO_ESTADO: datetime.fromtimestamp(estado.ultima_marca, tz=UTC),
        }
        for tarjeta, estado in estados.items()
    ]
    return pd.DataFrame(filas)


def _a_estado(valores: Mapping[str, Any], posicion: int) -> EstadoTarjeta | None:
    """Arma el estado de la fila `posicion`; `None` si Feast no tiene esa tarjeta."""
    if valores["cantidad_total"][posicion] is None:
        return None
    return EstadoTarjeta(
        cantidad_total=int(valores["cantidad_total"][posicion]),
        monto_total=float(valores["monto_total"][posicion]),
        categorias_vistas=tuple(valores["categorias_vistas"][posicion]),
        marcas_recientes=tuple(int(m) for m in valores["marcas_recientes"][posicion]),
        montos_recientes=tuple(float(m) for m in valores["montos_recientes"][posicion]),
        ultima_marca=int(valores["ultima_marca"][posicion]),
        ultimo_id_transaccion=str(valores["ultimo_id_transaccion"][posicion]),
        ultima_latitud_comercio=float(valores["ultima_latitud_comercio"][posicion]),
        ultima_longitud_comercio=float(valores["ultima_longitud_comercio"][posicion]),
    )


def escribir_estados(tienda: FeatureStore, estados: Mapping[int, EstadoTarjeta]) -> None:
    """Empuja el estado de las tarjetas al almacén online (`push`, solo online)."""
    if not estados:
        return
    tienda.push(NOMBRE_FUENTE_PUSH_ESTADO, estados_a_dataframe(estados), to=PushMode.ONLINE)


def leer_estados(tienda: FeatureStore, tarjetas: Sequence[int]) -> dict[int, EstadoTarjeta | None]:
    """Lee el estado de cada tarjeta; `None` para las que todavía no tienen historia."""
    if not tarjetas:
        return {}
    respuesta = tienda.get_online_features(
        features=[f"{NOMBRE_VISTA_ESTADO_TARJETA}:{campo}" for campo in CAMPOS_ESTADO],
        entity_rows=[{"numero_tarjeta": tarjeta} for tarjeta in tarjetas],
    ).to_dict()
    return {tarjeta: _a_estado(respuesta, i) for i, tarjeta in enumerate(tarjetas)}
