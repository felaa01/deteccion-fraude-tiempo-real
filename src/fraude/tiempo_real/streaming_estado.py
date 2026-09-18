"""Job de Spark Structured Streaming: mantiene en Redis el estado por tarjeta.

Lee las transacciones del tópico de Kafka y, por cada micro-lote (`foreachBatch`), actualiza el
estado de las tarjetas involucradas en el almacén online de Feast (ver
`fraude.tiempo_real.lote_estado`). Spark aporta el consumo tolerante a fallas de Kafka: los
offsets ya procesados quedan en un checkpoint, así que tras un reinicio retoma donde quedó.

Uso: `make streaming` (necesita `make tiempo-real-arriba`, el tópico creado y el arranque en
frío hecho). Con `--hasta-agotar` procesa lo que hay en el tópico y termina; sin él, queda
corriendo esperando transacciones nuevas.
"""

import argparse
import logging
from pathlib import Path
from typing import Any

from feast import FeatureStore
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.streaming.query import StreamingQuery
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

from fraude.caracteristicas.arranque_en_frio import RUTA_REGISTRO
from fraude.caracteristicas.estado_tarjeta import TransaccionTarjeta
from fraude.caracteristicas.tienda import crear_tienda_online
from fraude.lotes.comun import crear_sesion_spark
from fraude.tiempo_real.lote_estado import TransaccionEntrante, procesar_lote

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Conector de Kafka para Spark: la versión de Spark y la de Scala (2.13) tienen que coincidir
# con las del PySpark instalado. Spark lo baja de Maven Central la primera vez (gratis).
PAQUETE_KAFKA = "org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0"
RUTA_CHECKPOINT = Path("datos_features/checkpoint_streaming")

# Solo los campos que usa el estado; el resto del mensaje se ignora.
ESQUEMA_MENSAJE = StructType(
    [
        StructField("id_transaccion", StringType()),
        StructField("numero_tarjeta", LongType()),
        StructField("fecha_hora_transaccion", StringType()),
        StructField("monto", DoubleType()),
        StructField("categoria", StringType()),
    ]
)


def parsear_mensajes(mensajes: DataFrame) -> DataFrame:
    """Convierte la columna binaria `value` (JSON) en columnas tipadas.

    Un mensaje que no es JSON válido queda con todas las columnas nulas en vez de romper el
    lote; el llamador los descarta y los cuenta.

    La fecha viaja sin zona horaria ("2020-06-21T12:14:25") y se interpreta como UTC de forma
    **explícita** (se le agrega `Z` antes de parsear) en vez de depender de la zona horaria de
    la sesión de Spark, que es exactamente el bug que corrompió los timestamps en la semana 3.
    """
    return (
        mensajes.select(
            F.from_json(F.col("value").cast("string"), ESQUEMA_MENSAJE).alias("mensaje")
        )
        .select("mensaje.*")
        .withColumn(
            "marca_tiempo",
            F.to_timestamp(
                F.concat(F.col("fecha_hora_transaccion"), F.lit("Z")), "yyyy-MM-dd'T'HH:mm:ssX"
            ).cast("long"),
        )
        .select("numero_tarjeta", "id_transaccion", "marca_tiempo", "monto", "categoria")
    )


def a_transacciones_entrantes(
    filas: list[tuple[Any, ...]],
) -> tuple[list[TransaccionEntrante], int]:
    """Filas de `parsear_mensajes` a transacciones; devuelve también cuántas eran inválidas."""
    validas: list[TransaccionEntrante] = []
    for fila in filas:
        tarjeta, id_transaccion, marca, monto, categoria = fila
        if None in fila:
            continue
        validas.append(
            TransaccionEntrante(
                int(tarjeta),
                TransaccionTarjeta(str(id_transaccion), int(marca), float(monto), str(categoria)),
            )
        )
    return validas, len(filas) - len(validas)


def _procesar_lote_de_spark(tienda: FeatureStore, lote: DataFrame, id_lote: int) -> None:
    filas = [tuple(fila) for fila in lote.collect()]
    entrantes, invalidas = a_transacciones_entrantes(filas)
    if invalidas:
        logger.warning("Lote %d: %d mensajes inválidos descartados", id_lote, invalidas)
    if not entrantes:
        return
    resumen = procesar_lote(tienda, entrantes)
    logger.info(
        "Lote %d: %d transacciones, %d tarjetas, %d actualizadas",
        id_lote,
        resumen.transacciones,
        resumen.tarjetas,
        resumen.tarjetas_actualizadas,
    )


def iniciar(
    servidor: str,
    topico: str,
    checkpoint: Path,
    tienda: FeatureStore,
    hasta_agotar: bool,
    intervalo_segundos: int,
    maximo_por_lote: int,
) -> StreamingQuery:
    # 1 GB de heap para el driver: el job corre junto con Kafka (1 GB) y Redis (256 MB) dentro
    # de los 5 GB de WSL, y solo procesa lotes chicos.
    spark = crear_sesion_spark("fraude-streaming-estado", [PAQUETE_KAFKA], memoria_driver="1g")
    mensajes = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", servidor)
        .option("subscribe", topico)
        .option("startingOffsets", "earliest")
        # Acota el tamaño de cada lote (y con él la memoria del driver) cuando hay mucho
        # atrasado en el tópico.
        .option("maxOffsetsPerTrigger", maximo_por_lote)
        .load()
    )
    escritura = (
        parsear_mensajes(mensajes)
        .writeStream.foreachBatch(
            lambda lote, id_lote: _procesar_lote_de_spark(tienda, lote, id_lote)
        )
        .option("checkpointLocation", str(checkpoint))
    )
    if hasta_agotar:
        escritura = escritura.trigger(availableNow=True)
    else:
        escritura = escritura.trigger(processingTime=f"{intervalo_segundos} seconds")
    return escritura.start()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--servidor", default="localhost:9092")
    parser.add_argument("--topico", default="transacciones")
    parser.add_argument("--checkpoint", type=Path, default=RUTA_CHECKPOINT)
    parser.add_argument("--registro", type=Path, default=RUTA_REGISTRO)
    parser.add_argument("--conexion-redis", default="localhost:6379")
    parser.add_argument("--proyecto", default="deteccion_fraude")
    parser.add_argument("--hasta-agotar", action="store_true")
    parser.add_argument("--intervalo", type=int, default=2, help="Segundos entre lotes.")
    parser.add_argument("--maximo-por-lote", type=int, default=10_000)
    args = parser.parse_args()

    tienda = crear_tienda_online(args.registro, args.conexion_redis, args.proyecto)
    consulta = iniciar(
        args.servidor,
        args.topico,
        args.checkpoint,
        tienda,
        args.hasta_agotar,
        args.intervalo,
        args.maximo_por_lote,
    )
    consulta.awaitTermination()


if __name__ == "__main__":
    main()
