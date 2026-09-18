import json
from datetime import datetime

import pandas as pd
import pytest

from fraude.productor.mensajes import (
    CAMPOS_MENSAJE,
    clave_de_particion,
    iterar_filas,
    preparar_transacciones,
    serializar,
)
from fraude.productor.ritmo import instante_de_envio


def _transacciones() -> pd.DataFrame:
    """Tres transacciones desordenadas, dos con la misma fecha, con columnas de más."""
    return pd.DataFrame(
        {
            "id_transaccion": ["c", "b", "a"],
            "numero_tarjeta": [222, 111, 111],
            "fecha_hora_transaccion": pd.to_datetime(
                ["2020-06-21 12:00:05", "2020-06-21 12:00:00", "2020-06-21 12:00:00"]
            ),
            "monto": [10.5, 20.0, 30.0],
            "categoria": ["ocio", "hogar", "hogar"],
            "comercio": ["x", "y", "z"],
            "genero": ["F", "M", "M"],
            "fecha_nacimiento": pd.to_datetime(["1980-01-02", "1970-03-04", "1970-03-04"]),
            "latitud_cliente": [1.0, 2.0, 3.0],
            "longitud_cliente": [1.0, 2.0, 3.0],
            "latitud_comercio": [1.5, 2.5, 3.5],
            "longitud_comercio": [1.5, 2.5, 3.5],
            "poblacion_ciudad": [100, 200, 300],
            # No deben viajar en el mensaje.
            "es_fraude": [0, 1, 0],
            "nombre": ["Ana", "Beto", "Carla"],
            "marca_tiempo_unix": [1, 2, 3],
        }
    )


def prueba_preparar_ordena_por_fecha_y_desempata_por_id() -> None:
    preparadas = preparar_transacciones(_transacciones())

    assert preparadas["id_transaccion"].tolist() == ["a", "b", "c"]


def prueba_el_mensaje_no_incluye_etiqueta_ni_datos_personales() -> None:
    preparadas = preparar_transacciones(_transacciones())

    assert list(preparadas.columns) == list(CAMPOS_MENSAJE)
    for excluido in ("es_fraude", "nombre", "marca_tiempo_unix"):
        assert excluido not in preparadas.columns


def prueba_serializar_produce_json_con_fechas_iso() -> None:
    fila = next(iterar_filas(preparar_transacciones(_transacciones())))

    mensaje = json.loads(serializar(fila))

    assert mensaje["id_transaccion"] == "a"
    assert mensaje["fecha_hora_transaccion"] == "2020-06-21T12:00:00"
    assert mensaje["fecha_nacimiento"] == "1970-03-04T00:00:00"
    assert mensaje["monto"] == 30.0
    assert mensaje["numero_tarjeta"] == 111


def prueba_serializar_rechaza_tipos_desconocidos() -> None:
    with pytest.raises(TypeError):
        serializar({"raro": object()})


def prueba_misma_tarjeta_misma_clave() -> None:
    filas = list(iterar_filas(preparar_transacciones(_transacciones())))

    claves = [clave_de_particion(fila) for fila in filas]

    assert claves == [b"111", b"111", b"222"]


def prueba_iterar_filas_por_bloques_conserva_todas_en_orden() -> None:
    preparadas = preparar_transacciones(_transacciones())

    filas = list(iterar_filas(preparadas, tamano_bloque=2))

    assert [fila["id_transaccion"] for fila in filas] == ["a", "b", "c"]


def prueba_instante_de_envio_escala_por_la_aceleracion() -> None:
    inicial = datetime(2020, 6, 21, 12, 0, 0)

    # 1 hora simulada a x3600 son 1 segundo reales.
    objetivo = instante_de_envio(datetime(2020, 6, 21, 13, 0, 0), inicial, 100.0, 3600.0)

    assert objetivo == pytest.approx(101.0)


def prueba_instante_de_envio_del_primer_evento_es_el_inicio() -> None:
    inicial = datetime(2020, 6, 21, 12, 0, 0)

    assert instante_de_envio(inicial, inicial, 100.0, 60.0) == 100.0


@pytest.mark.parametrize("aceleracion", [0.0, -1.0])
def prueba_instante_de_envio_rechaza_aceleracion_no_positiva(aceleracion: float) -> None:
    fecha = datetime(2020, 6, 21)

    with pytest.raises(ValueError, match="mayor que cero"):
        instante_de_envio(fecha, fecha, 0.0, aceleracion)
