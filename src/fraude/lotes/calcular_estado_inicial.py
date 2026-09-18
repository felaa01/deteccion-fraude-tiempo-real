"""Calcula el estado por tarjeta al final de `fraudTrain.csv` y lo guarda en Parquet.

Es el punto de partida del almacén online: `fraude.caracteristicas.arranque_en_frio` lo carga
en Redis con `materialize`, y después el streaming solo aplica lo que llega desde
`fraudTest.csv`. Se usa solo `fraudTrain`: es lo que un sistema en producción "ya sabía" antes
de empezar a recibir transacciones nuevas.
"""

import logging
from pathlib import Path

from fraude.lotes.comun import cargar_transacciones, crear_sesion_spark
from fraude.lotes.estado_inicial import calcular_estado_tarjeta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RUTA_ENTRADA = Path("datos/fraudTrain.csv")
RUTA_SALIDA = Path("datos_features/estado_tarjeta_inicial.parquet")


def main() -> None:
    spark = crear_sesion_spark("fraude-estado-inicial")
    try:
        transacciones = cargar_transacciones(spark, [RUTA_ENTRADA])
        estado = calcular_estado_tarjeta(transacciones)
        # Una fila por tarjeta (~1.000): un solo archivo alcanza y es más fácil de inspeccionar.
        estado.coalesce(1).write.mode("overwrite").parquet(str(RUTA_SALIDA))
        logger.info("Estado inicial de %d tarjetas escrito en %s", estado.count(), RUTA_SALIDA)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
