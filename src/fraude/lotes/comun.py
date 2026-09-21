"""Piezas compartidas por los jobs por lotes de PySpark."""

from collections.abc import Sequence
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from fraude.entrenamiento.carga import COLUMNAS_EN_ESPANOL


def crear_sesion_spark(
    nombre: str, paquetes: Sequence[str] = (), memoria_driver: str = "2g"
) -> SparkSession:
    """Sesión local de Spark. `paquetes` son coordenadas Maven (ej. el conector de Kafka).

    Los paquetes solo se aplican si la sesión se crea en este llamado: `getOrCreate` devuelve
    la sesión existente, ignorando la configuración nueva, por eso los jobs que los necesitan
    corren en su propio proceso.
    """
    # `local[4]`, no `local[*]`: máquina de 8 GB (WSL limitado a 5 GB, sin GPU), y este
    # job corre junto con lo demás que uno tenga abierto -- no hace falta acaparar todos
    # los núcleos para un dataset de ~1,8 millones de filas.
    constructor = (
        SparkSession.builder.appName(nombre)
        .master("local[4]")
        .config("spark.driver.memory", memoria_driver)
        .config("spark.sql.shuffle.partitions", "8")
    )
    if paquetes:
        constructor = constructor.config("spark.jars.packages", ",".join(paquetes))
    return constructor.getOrCreate()


def cargar_transacciones(spark: SparkSession, rutas_csv: list[Path]) -> DataFrame:
    """Lee los CSV crudos, los renombra al español y castea las columnas que se usan.

    Devuelve, además de las columnas del dataset, `marca_tiempo` (segundos desde la época, UTC).

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
        F.col("latitud_comercio").cast("double").alias("latitud_comercio"),
        F.col("longitud_comercio").cast("double").alias("longitud_comercio"),
        F.to_timestamp("fecha_hora_transaccion", "yyyy-MM-dd HH:mm:ss").alias(
            "fecha_hora_transaccion"
        ),
    ).withColumn(
        # Segundos desde la época, derivados de la FECHA (no de `unix_time`). En `fraudTrain`
        # el desfase entre las dos columnas no es constante (2557 días, 2556 entre el
        # 2019-02-28 y el 2020-03-01) y el archivo está ordenado por `unix_time` pero no por
        # fecha; el streaming y el servicio solo ven la fecha, así que el batch tiene que
        # ordenar y ventanear con la misma definición de tiempo para no producir skew.
        "marca_tiempo",
        F.col("fecha_hora_transaccion").cast("long"),
    )
