"""Calcula las características históricas por tarjeta sobre todo el dataset y las
guarda en Parquet, listas para que Feast las use como almacén offline.

Corre sobre `fraudTrain.csv` + `fraudTest.csv` concatenados: la historia de cada tarjeta
es continua más allá del corte de train/test que se usa para el modelado, así que hay que
verla completa para no arrancar la ventana de cada tarjeta a mitad de su historia real.
Esto no filtra información del futuro hacia el pasado porque cada fila solo mira, hacia
atrás, transacciones anteriores de su propia tarjeta (ver
`fraude.lotes.caracteristicas_historicas`).
"""

import logging
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from fraude.entrenamiento.carga import COLUMNAS_EN_ESPANOL
from fraude.lotes.caracteristicas_historicas import agregar_caracteristicas_historicas

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RUTA_DATOS = Path("datos")
RUTA_SALIDA = Path("datos_features/historico_tarjeta.parquet")

# Columnas que necesita el cálculo de características más el timestamp que va a usar
# Feast como `event_timestamp`. El resto de las columnas del dataset (nombre, dirección,
# coordenadas, etc.) no hacen falta para este job.
COLUMNAS_DE_SALIDA = [
    "numero_tarjeta",
    "fecha_hora_transaccion",
    "id_transaccion",
    "cantidad_transacciones_10min",
    "cantidad_transacciones_1h",
    "cantidad_transacciones_24h",
    "monto_acumulado_tarjeta",
    "ratio_monto_promedio_tarjeta",
    "categoria_nueva_para_tarjeta",
]


def _crear_sesion_spark() -> SparkSession:
    # `local[4]`, no `local[*]`: máquina de 8 GB (WSL limitado a 5 GB, sin GPU), y este
    # job corre junto con lo demás que uno tenga abierto -- no hace falta acaparar todos
    # los núcleos para un dataset de ~1,8 millones de filas.
    return (
        SparkSession.builder.appName("fraude-caracteristicas-historicas")
        .master("local[4]")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def _cargar_transacciones(spark: SparkSession, rutas_csv: list[Path]) -> DataFrame:
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


def main() -> None:
    spark = _crear_sesion_spark()
    try:
        transacciones = _cargar_transacciones(
            spark, [RUTA_DATOS / "fraudTrain.csv", RUTA_DATOS / "fraudTest.csv"]
        )
        con_caracteristicas = agregar_caracteristicas_historicas(transacciones)
        con_caracteristicas.select(COLUMNAS_DE_SALIDA).write.mode("overwrite").parquet(
            str(RUTA_SALIDA)
        )
        logger.info("Características históricas escritas en %s", RUTA_SALIDA)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
