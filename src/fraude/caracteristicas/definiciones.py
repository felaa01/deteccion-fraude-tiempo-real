"""Definiciones de Feast para las características históricas de tarjeta.

Almacén offline: Parquet (`FileSource`), generado por `fraude.lotes.calcular_historico`.
Todavía no hay almacén online real (llega en la semana 4, con el push a Redis desde Spark
Streaming): por eso `online=False` en la vista, esta primera version solo sirve para el
join point-in-time al armar datasets de entrenamiento.
"""

from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource, Project
from feast.types import Bool, Float64, Int64
from feast.value_type import ValueType

NOMBRE_VISTA_CARACTERISTICAS_HISTORICAS = "caracteristicas_historicas_tarjeta"

PROYECTO = Project(
    name="deteccion_fraude",
    description="Deteccion de fraude con tarjetas en tiempo real.",
)

TARJETA = Entity(name="tarjeta", join_keys=["numero_tarjeta"], value_type=ValueType.INT64)


def crear_fuente_historico(ruta: Path) -> FileSource:
    """Fuente de Feast para el Parquet con las características históricas por tarjeta."""
    return FileSource(
        name="historico_tarjeta_source",
        path=str(ruta),
        timestamp_field="fecha_hora_transaccion",
    )


def crear_vista_caracteristicas_historicas(fuente: FileSource) -> FeatureView:
    """Vista de Feast con el esquema de `fraude.lotes.caracteristicas_historicas`."""
    return FeatureView(
        name=NOMBRE_VISTA_CARACTERISTICAS_HISTORICAS,
        entities=[TARJETA],
        online=False,
        schema=[
            Field(name="cantidad_transacciones_10min", dtype=Int64),
            Field(name="cantidad_transacciones_1h", dtype=Int64),
            Field(name="cantidad_transacciones_24h", dtype=Int64),
            Field(name="monto_acumulado_tarjeta", dtype=Float64),
            Field(name="ratio_monto_promedio_tarjeta", dtype=Float64),
            Field(name="categoria_nueva_para_tarjeta", dtype=Bool),
        ],
        source=fuente,
    )
