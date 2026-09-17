import pandas as pd
import pytest
from pyspark.sql import Row, SparkSession
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

from fraude.lotes.caracteristicas_historicas import agregar_caracteristicas_historicas

ESQUEMA = StructType(
    [
        StructField("id_transaccion", StringType()),
        StructField("numero_tarjeta", LongType()),
        StructField("marca_tiempo_unix", LongType()),
        StructField("categoria", StringType()),
        StructField("monto", DoubleType()),
    ]
)

# Tarjeta 111: tx_a (t=0, comida) -> tx_b (5 min después, comida) -> tx_c (20 min después
# de tx_a, 15 min después de tx_b, ropa). Tarjeta 222: una sola transacción, sin relación
# con la 111.
FILAS = [
    Row(
        id_transaccion="tx_a",
        numero_tarjeta=111,
        marca_tiempo_unix=0,
        categoria="comida",
        monto=10.0,
    ),
    Row(
        id_transaccion="tx_b",
        numero_tarjeta=111,
        marca_tiempo_unix=300,
        categoria="comida",
        monto=20.0,
    ),
    Row(
        id_transaccion="tx_c",
        numero_tarjeta=111,
        marca_tiempo_unix=1200,
        categoria="ropa",
        monto=30.0,
    ),
    Row(
        id_transaccion="tx_d",
        numero_tarjeta=222,
        marca_tiempo_unix=100,
        categoria="comida",
        monto=5.0,
    ),
]


def _calcular(sesion_spark: SparkSession) -> pd.DataFrame:
    transacciones = sesion_spark.createDataFrame(FILAS, ESQUEMA)
    resultado = agregar_caracteristicas_historicas(transacciones).toPandas()
    return resultado.set_index("id_transaccion")


def prueba_primera_transaccion_de_una_tarjeta_no_tiene_historia(
    sesion_spark: SparkSession,
) -> None:
    fila = _calcular(sesion_spark).loc["tx_a"]

    assert fila["cantidad_transacciones_10min"] == 0
    assert fila["cantidad_transacciones_1h"] == 0
    assert fila["cantidad_transacciones_24h"] == 0
    assert pd.isna(fila["monto_acumulado_tarjeta"])
    assert pd.isna(fila["ratio_monto_promedio_tarjeta"])
    assert fila["categoria_nueva_para_tarjeta"]


def prueba_no_ve_transacciones_futuras_de_la_misma_tarjeta(
    sesion_spark: SparkSession,
) -> None:
    # tx_a es anterior a tx_b y tx_c: si alguna ventana de tx_a las contara, habría fuga
    # de información del futuro.
    fila = _calcular(sesion_spark).loc["tx_a"]

    assert fila["cantidad_transacciones_24h"] == 0


def prueba_cuenta_transacciones_previas_dentro_de_la_ventana(
    sesion_spark: SparkSession,
) -> None:
    fila = _calcular(sesion_spark).loc["tx_b"]

    # tx_a fue 5 minutos antes: entra en la ventana de 10 minutos, 1 hora y 24 horas.
    assert fila["cantidad_transacciones_10min"] == 1
    assert fila["cantidad_transacciones_1h"] == 1
    assert fila["cantidad_transacciones_24h"] == 1
    assert fila["monto_acumulado_tarjeta"] == 10.0
    assert fila["ratio_monto_promedio_tarjeta"] == pytest.approx(2.0)


def prueba_transaccion_previa_fuera_de_ventana_corta_no_cuenta(
    sesion_spark: SparkSession,
) -> None:
    fila = _calcular(sesion_spark).loc["tx_c"]

    # tx_b fue 15 minutos antes: no entra en la ventana de 10 minutos, sí en 1h/24h.
    assert fila["cantidad_transacciones_10min"] == 0
    assert fila["cantidad_transacciones_1h"] == 2
    assert fila["cantidad_transacciones_24h"] == 2
    assert fila["monto_acumulado_tarjeta"] == 30.0
    assert fila["ratio_monto_promedio_tarjeta"] == pytest.approx(2.0)


def prueba_categoria_nueva_solo_en_la_primera_aparicion_por_tarjeta(
    sesion_spark: SparkSession,
) -> None:
    resultado = _calcular(sesion_spark)

    assert resultado.loc["tx_a", "categoria_nueva_para_tarjeta"]  # comida, 1ra vez
    assert not resultado.loc["tx_b", "categoria_nueva_para_tarjeta"]  # comida, repetida
    assert resultado.loc["tx_c", "categoria_nueva_para_tarjeta"]  # ropa, 1ra vez


def prueba_no_mezcla_historia_entre_tarjetas_distintas(
    sesion_spark: SparkSession,
) -> None:
    fila = _calcular(sesion_spark).loc["tx_d"]

    assert fila["cantidad_transacciones_24h"] == 0
    assert pd.isna(fila["monto_acumulado_tarjeta"])
