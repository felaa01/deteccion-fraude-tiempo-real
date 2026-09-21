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

from fraude.lotes.caracteristicas_historicas import agregar_caracteristicas_historicas
from fraude.lotes.comun import cargar_transacciones, crear_sesion_spark

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
    "distancia_transaccion_anterior_km",
    "velocidad_implicita_kmh",
]


def main() -> None:
    spark = crear_sesion_spark("fraude-caracteristicas-historicas")
    try:
        transacciones = cargar_transacciones(
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
