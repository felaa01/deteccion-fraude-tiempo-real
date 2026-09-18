"""Estado por tarjeta al corte de un histórico, para el arranque en frío del almacén online.

Es la segunda implementación, independiente y hecha con agregaciones de Spark, de lo que el
streaming calcula transacción a transacción con `fraude.caracteristicas.estado_tarjeta.
actualizar_estado`. Que las dos coincidan sobre los mismos datos (ver
`pruebas/prueba_estado_inicial.py`) es la evidencia de que el lote y el streaming dan lo
mismo, en vez de confiar en que una sola implementación no tenga un bug.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from fraude.caracteristicas.estado_tarjeta import SEGUNDOS_24_HORAS


def calcular_estado_tarjeta(transacciones: DataFrame) -> DataFrame:
    """Una fila por tarjeta con las columnas de `EstadoTarjeta` y `fecha_hora_ultima_transaccion`.

    Requiere `numero_tarjeta`, `id_transaccion`, `categoria`, `monto` y
    `fecha_hora_transaccion` (timestamp). El tiempo se toma de `fecha_hora_transaccion`, no de
    `marca_tiempo_unix`, que en este dataset está desfasada 7 años.
    """
    con_marca = transacciones.withColumn(
        "marca_tiempo", F.col("fecha_hora_transaccion").cast("long")
    )

    # La "última" transacción es la mayor por (marca, id), el mismo orden con el que el
    # streaming decide qué ya aplicó; `max` sobre un struct compara campo por campo.
    resumen = con_marca.groupBy("numero_tarjeta").agg(
        F.count("*").alias("cantidad_total"),
        F.sum("monto").alias("monto_total"),
        F.array_sort(F.collect_set("categoria")).alias("categorias_vistas"),
        F.max(F.struct("marca_tiempo", "id_transaccion")).alias("ultima"),
    )
    resumen = resumen.select(
        "numero_tarjeta",
        "cantidad_total",
        "monto_total",
        "categorias_vistas",
        F.col("ultima.marca_tiempo").alias("ultima_marca"),
        F.col("ultima.id_transaccion").alias("ultimo_id_transaccion"),
    )

    # Recientes: todo lo que cae a menos de 24 h de la última transacción de la tarjeta
    # (equivale a lo que `actualizar_estado` conserva tras podar en cada paso). El struct se
    # ordena por (marca, id), que es el orden en que el streaming las va agregando.
    recientes = (
        con_marca.join(resumen.select("numero_tarjeta", "ultima_marca"), "numero_tarjeta")
        .where(F.col("marca_tiempo") >= F.col("ultima_marca") - SEGUNDOS_24_HORAS)
        .groupBy("numero_tarjeta")
        .agg(
            F.array_sort(F.collect_list(F.struct("marca_tiempo", "id_transaccion", "monto"))).alias(
                "recientes"
            )
        )
        .select(
            "numero_tarjeta",
            F.transform("recientes", lambda r: r["marca_tiempo"]).alias("marcas_recientes"),
            F.transform("recientes", lambda r: r["monto"]).alias("montos_recientes"),
        )
    )

    return resumen.join(recientes, "numero_tarjeta").select(
        "numero_tarjeta",
        "cantidad_total",
        "monto_total",
        "categorias_vistas",
        "marcas_recientes",
        "montos_recientes",
        "ultima_marca",
        "ultimo_id_transaccion",
        F.col("ultima_marca").cast("timestamp").alias("fecha_hora_ultima_transaccion"),
    )
