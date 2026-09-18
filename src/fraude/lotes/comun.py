"""Piezas compartidas por los jobs por lotes de PySpark."""

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from fraude.entrenamiento.carga import COLUMNAS_EN_ESPANOL


def crear_sesion_spark(nombre: str) -> SparkSession:
    # `local[4]`, no `local[*]`: máquina de 8 GB (WSL limitado a 5 GB, sin GPU), y este
    # job corre junto con lo demás que uno tenga abierto -- no hace falta acaparar todos
    # los núcleos para un dataset de ~1,8 millones de filas.
    return (
        SparkSession.builder.appName(nombre)
        .master("local[4]")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def cargar_transacciones(spark: SparkSession, rutas_csv: list[Path]) -> DataFrame:
    """Lee los CSV crudos, los renombra al español y castea las columnas que se usan.

    Se lee con `inferSchema=False` (todo como string) y se castea a mano en vez de
    confiar en la inferencia de tipos de Spark: es explícito y evita una pasada extra
    de Spark sobre todo el archivo solo para adivinar tipos.
    """
    # `to_timestamp` interpreta el string con la zona horaria de sesión (que por
    # default toma la del sistema). Sin fijarla a UTC, esta máquina (UTC-3) correría
    # todos los timestamps 3 horas y rompería en silencio -- sin ningún error -- el
    # join point-in-time contra entity_df armados en pandas, que toman el string tal
    # cual, sin huso horario.
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    crudo = spark.read.csv([str(ruta) for ruta in rutas_csv], header=True, inferSchema=False)
    for original, espanol in COLUMNAS_EN_ESPANOL.items():
        crudo = crudo.withColumnRenamed(original, espanol)

    return crudo.select(
        F.col("numero_tarjeta").cast("long").alias("numero_tarjeta"),
        "categoria",
        "id_transaccion",
        F.col("monto").cast("double").alias("monto"),
        F.col("marca_tiempo_unix").cast("long").alias("marca_tiempo_unix"),
        F.to_timestamp("fecha_hora_transaccion", "yyyy-MM-dd HH:mm:ss").alias(
            "fecha_hora_transaccion"
        ),
    )
