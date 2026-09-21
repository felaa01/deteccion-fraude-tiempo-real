"""Carga de las transacciones crudas del dataset de Kaggle (Sparkov) y renombre al español."""

from pathlib import Path

import pandas as pd

# Mapeo de las columnas originales del dataset de Kaggle a nombres en español.
# Los nombres originales son una excepción inevitable: son las columnas tal como
# vienen en el CSV, y se renombran acá, en el único lugar donde se cargan.
COLUMNAS_EN_ESPANOL = {
    "trans_date_trans_time": "fecha_hora_transaccion",
    "cc_num": "numero_tarjeta",
    "merchant": "comercio",
    "category": "categoria",
    "amt": "monto",
    "first": "nombre",
    "last": "apellido",
    "gender": "genero",
    "street": "calle",
    "city": "ciudad",
    "state": "estado",
    "zip": "codigo_postal",
    "lat": "latitud_cliente",
    "long": "longitud_cliente",
    "city_pop": "poblacion_ciudad",
    "job": "ocupacion",
    "dob": "fecha_nacimiento",
    "trans_num": "id_transaccion",
    "unix_time": "marca_tiempo_unix",
    "merch_lat": "latitud_comercio",
    "merch_long": "longitud_comercio",
    "is_fraud": "es_fraude",
}


def cargar_transacciones(ruta: Path) -> pd.DataFrame:
    """Carga un CSV crudo del dataset y devuelve un DataFrame con columnas en español.

    La primera columna del CSV es un índice sin nombre que dejó el generador del
    dataset al exportarlo; se descarta con `index_col=0`.
    """
    df = pd.read_csv(
        ruta,
        index_col=0,
        # El parser rápido por defecto de pandas no siempre redondea bien el último decimal
        # (43.274585 puede leerse como 43.274584999999995). `round_trip` da el mismo double
        # que el `cast` de Spark, así el estado batch y el del streaming son idénticos bit a bit.
        float_precision="round_trip",
        parse_dates=["trans_date_trans_time", "dob"],
    )
    return df.rename(columns=COLUMNAS_EN_ESPANOL)
