from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession

from fraude.lotes.calcular_historico import _cargar_transacciones

ENCABEZADO = (
    ",trans_date_trans_time,cc_num,merchant,category,amt,first,last,gender,street,"
    "city,state,zip,lat,long,city_pop,job,dob,trans_num,unix_time,merch_lat,"
    "merch_long,is_fraud"
)
FILA = (
    '0,2019-01-01 00:00:18,2703186189652095,"fraud_Rippin, Kub and Mann",misc_net,4.97,'
    "Jennifer,Banks,F,561 Perry Cove,Moravian Falls,NC,28654,36.0788,-81.1781,3495,"
    '"Psychologist, counselling",1988-03-09,0b242abb623afc578575680df30655b9,'
    "1325376018,36.011293,-82.048315,0"
)


def prueba_no_corre_los_timestamps_por_la_zona_horaria_del_sistema(
    sesion_spark: SparkSession, tmp_path: Path
) -> None:
    # Regresión: `to_timestamp` toma la zona horaria de sesión de Spark si no se fija
    # explícitamente a UTC. En una máquina en UTC-3, esto no se nota si se lee de
    # vuelta con el propio `toPandas()` de Spark (la misma conversión de zona horaria
    # se cancela sola de ida y de vuelta) -- pero sí corrompe el valor 3 horas cuando
    # el Parquet lo lee después otra herramienta que no conoce esa zona horaria de
    # sesión, como pandas/pyarrow (que es exactamente como Feast lee el almacén
    # offline). Por eso esta prueba escribe a Parquet real y lo relee con pandas, en
    # vez de comparar contra el propio `toPandas()` de la misma sesión de Spark.
    ruta_csv = tmp_path / "fraudTrain.csv"
    ruta_csv.write_text(ENCABEZADO + "\n" + FILA + "\n")
    ruta_parquet = tmp_path / "salida.parquet"

    transacciones = _cargar_transacciones(sesion_spark, [ruta_csv])
    transacciones.write.mode("overwrite").parquet(str(ruta_parquet))
    resultado = pd.read_parquet(ruta_parquet)

    assert resultado.loc[0, "fecha_hora_transaccion"] == pd.Timestamp("2019-01-01 00:00:18")
    assert resultado.loc[0, "marca_tiempo_unix"] == 1325376018
