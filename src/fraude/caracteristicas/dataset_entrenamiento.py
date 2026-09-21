"""Armado de datasets de entrenamiento con el join point-in-time de Feast."""

from pathlib import Path

import pandas as pd
from feast import FeatureStore

from fraude.caracteristicas.definiciones import (
    NOMBRE_VISTA_CARACTERISTICAS_HISTORICAS,
    PROYECTO,
    TARJETA,
    crear_fuente_historico,
    crear_vista_caracteristicas_historicas,
)

CARACTERISTICAS_HISTORICAS = [
    f"{NOMBRE_VISTA_CARACTERISTICAS_HISTORICAS}:cantidad_transacciones_10min",
    f"{NOMBRE_VISTA_CARACTERISTICAS_HISTORICAS}:cantidad_transacciones_1h",
    f"{NOMBRE_VISTA_CARACTERISTICAS_HISTORICAS}:cantidad_transacciones_24h",
    f"{NOMBRE_VISTA_CARACTERISTICAS_HISTORICAS}:monto_acumulado_tarjeta",
    f"{NOMBRE_VISTA_CARACTERISTICAS_HISTORICAS}:ratio_monto_promedio_tarjeta",
    f"{NOMBRE_VISTA_CARACTERISTICAS_HISTORICAS}:categoria_nueva_para_tarjeta",
    f"{NOMBRE_VISTA_CARACTERISTICAS_HISTORICAS}:distancia_transaccion_anterior_km",
    f"{NOMBRE_VISTA_CARACTERISTICAS_HISTORICAS}:velocidad_implicita_kmh",
]


def aplicar_definiciones(tienda: FeatureStore, ruta_historico: Path) -> None:
    """Da de alta en `tienda` el proyecto, la entidad y la vista de características."""
    fuente = crear_fuente_historico(ruta_historico)
    vista = crear_vista_caracteristicas_historicas(fuente)
    tienda.apply([PROYECTO, TARJETA, vista])


def agregar_caracteristicas_historicas_point_in_time(
    tienda: FeatureStore, transacciones: pd.DataFrame
) -> pd.DataFrame:
    """Suma a `transacciones` sus características históricas con un join point-in-time.

    `transacciones` necesita `numero_tarjeta` y `fecha_hora_transaccion`; el resto de sus
    columnas (monto, categoria, es_fraude, etc.) se preservan tal cual. `event_timestamp`
    es el nombre de columna que Feast espera para la marca de tiempo de la entidad; se
    renombra ida y vuelta para no filtrar ese detalle de Feast al resto del proyecto.
    """
    entity_df = transacciones.rename(columns={"fecha_hora_transaccion": "event_timestamp"})
    resultado = tienda.get_historical_features(
        entity_df=entity_df, features=CARACTERISTICAS_HISTORICAS
    ).to_df()
    return resultado.rename(columns={"event_timestamp": "fecha_hora_transaccion"})
