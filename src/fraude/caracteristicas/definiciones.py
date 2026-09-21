"""Definiciones de Feast para las características de tarjeta.

Hay dos vistas con propósitos distintos:

- `caracteristicas_historicas_tarjeta`: los valores ya calculados por transacción, en Parquet
  (`FileSource`, generado por `fraude.lotes.calcular_historico`). Solo sirve para el join
  point-in-time al armar datasets de entrenamiento, por eso `online=False`.
- `estado_tarjeta`: el estado crudo por tarjeta (ver `fraude.caracteristicas.estado_tarjeta`)
  en el almacén online (Redis). El servicio lo lee al recibir un pedido y calcula ahí las
  ventanas. Se alimenta con un `PushSource` desde Spark Streaming y, para el arranque en
  frío, con `materialize` desde un Parquet.
"""

from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource, Project, PushSource
from feast.types import Array, Bool, Float64, Int64, String
from feast.value_type import ValueType

NOMBRE_VISTA_CARACTERISTICAS_HISTORICAS = "caracteristicas_historicas_tarjeta"
NOMBRE_VISTA_ESTADO_TARJETA = "estado_tarjeta"
NOMBRE_FUENTE_PUSH_ESTADO = "estado_tarjeta_push"
# Columna de tiempo del estado: la fecha de la última transacción aplicada.
COLUMNA_TIEMPO_ESTADO = "fecha_hora_ultima_transaccion"

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
            Field(name="distancia_transaccion_anterior_km", dtype=Float64),
            Field(name="velocidad_implicita_kmh", dtype=Float64),
        ],
        source=fuente,
    )


def crear_fuente_estado_inicial(ruta: Path) -> FileSource:
    """Parquet con el estado de cada tarjeta al corte de `fraudTrain`, para el arranque en frío."""
    return FileSource(
        name="estado_tarjeta_inicial_source",
        path=str(ruta),
        timestamp_field=COLUMNA_TIEMPO_ESTADO,
    )


def crear_vista_estado_tarjeta(fuente_inicial: FileSource) -> FeatureView:
    """Vista online con el estado por tarjeta, con el esquema de `EstadoTarjeta`.

    Sin `ttl`: el estado de una tarjeta no vence por antigüedad (el acumulado histórico
    sigue valiendo aunque la tarjeta lleve meses sin usarse). Con una `ttl`, Feast devolvería
    nulos para tarjetas inactivas y el servicio las trataría como nuevas.
    """
    fuente_push = PushSource(name=NOMBRE_FUENTE_PUSH_ESTADO, batch_source=fuente_inicial)
    return FeatureView(
        name=NOMBRE_VISTA_ESTADO_TARJETA,
        entities=[TARJETA],
        online=True,
        schema=[
            Field(name="cantidad_total", dtype=Int64),
            Field(name="monto_total", dtype=Float64),
            Field(name="categorias_vistas", dtype=Array(String)),
            Field(name="marcas_recientes", dtype=Array(Int64)),
            Field(name="montos_recientes", dtype=Array(Float64)),
            Field(name="ultima_marca", dtype=Int64),
            Field(name="ultimo_id_transaccion", dtype=String),
            Field(name="ultima_latitud_comercio", dtype=Float64),
            Field(name="ultima_longitud_comercio", dtype=Float64),
        ],
        source=fuente_push,
    )
