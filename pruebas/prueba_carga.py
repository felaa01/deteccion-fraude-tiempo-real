from pathlib import Path

from fraude.entrenamiento.carga import cargar_transacciones

_COLUMNAS_CRUDAS = [
    "",
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "category",
    "amt",
    "first",
    "last",
    "gender",
    "street",
    "city",
    "state",
    "zip",
    "lat",
    "long",
    "city_pop",
    "job",
    "dob",
    "trans_num",
    "unix_time",
    "merch_lat",
    "merch_long",
    "is_fraud",
]
_FILA_CRUDA = [
    "0",
    "2019-01-01 00:00:18",
    "2703186189652095",
    '"fraud_Rippin, Kub and Mann"',
    "misc_net",
    "4.97",
    "Jennifer",
    "Banks",
    "F",
    "561 Perry Cove",
    "Moravian Falls",
    "NC",
    "28654",
    "36.0788",
    "-81.1781",
    "3495",
    "Psychologist",
    "1988-03-09",
    "0b242abb623afc578575680df30655b9",
    "1325376018",
    "36.011293",
    "-82.048315",
    "0",
]
CSV_DE_EJEMPLO = ",".join(_COLUMNAS_CRUDAS) + "\n" + ",".join(_FILA_CRUDA) + "\n"


def prueba_renombra_columnas_al_espanol(tmp_path: Path) -> None:
    ruta = tmp_path / "ejemplo.csv"
    ruta.write_text(CSV_DE_EJEMPLO)

    transacciones = cargar_transacciones(ruta)

    assert list(transacciones.columns) == [
        "fecha_hora_transaccion",
        "numero_tarjeta",
        "comercio",
        "categoria",
        "monto",
        "nombre",
        "apellido",
        "genero",
        "calle",
        "ciudad",
        "estado",
        "codigo_postal",
        "latitud_cliente",
        "longitud_cliente",
        "poblacion_ciudad",
        "ocupacion",
        "fecha_nacimiento",
        "id_transaccion",
        "marca_tiempo_unix",
        "latitud_comercio",
        "longitud_comercio",
        "es_fraude",
    ]


def prueba_parsea_fechas(tmp_path: Path) -> None:
    ruta = tmp_path / "ejemplo.csv"
    ruta.write_text(CSV_DE_EJEMPLO)

    transacciones = cargar_transacciones(ruta)

    fila = transacciones.iloc[0]
    assert str(fila["fecha_hora_transaccion"]) == "2019-01-01 00:00:18"
    assert str(fila["fecha_nacimiento"].date()) == "1988-03-09"


def prueba_conserva_datos_originales(tmp_path: Path) -> None:
    ruta = tmp_path / "ejemplo.csv"
    ruta.write_text(CSV_DE_EJEMPLO)

    transacciones = cargar_transacciones(ruta)

    fila = transacciones.iloc[0]
    assert fila["comercio"] == "fraud_Rippin, Kub and Mann"
    assert fila["monto"] == 4.97
    assert fila["es_fraude"] == 0
