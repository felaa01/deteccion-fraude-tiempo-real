import pandas as pd
import pytest

from fraude.entrenamiento.caracteristicas_basicas import agregar_caracteristicas_basicas


def _transaccion(**cambios: object) -> pd.DataFrame:
    base = {
        "fecha_hora_transaccion": pd.Timestamp("2019-06-15 14:30:00"),
        "fecha_nacimiento": pd.Timestamp("1989-06-15"),
        "latitud_cliente": 0.0,
        "longitud_cliente": 0.0,
        "latitud_comercio": 0.0,
        "longitud_comercio": 0.0,
    }
    base.update(cambios)
    return pd.DataFrame([base])


def prueba_extrae_hora_y_dia_de_la_semana() -> None:
    resultado = agregar_caracteristicas_basicas(_transaccion())

    fila = resultado.iloc[0]
    assert fila["hora_del_dia"] == 14
    assert fila["dia_de_la_semana"] == 5  # sábado


def prueba_calcula_edad_del_titular() -> None:
    resultado = agregar_caracteristicas_basicas(_transaccion())

    assert resultado.iloc[0]["edad_titular"] == pytest.approx(30.0, abs=0.01)


def prueba_distancia_es_cero_en_el_mismo_punto() -> None:
    resultado = agregar_caracteristicas_basicas(_transaccion())

    assert resultado.iloc[0]["distancia_cliente_comercio_km"] == 0.0


def prueba_distancia_entre_buenos_aires_y_montevideo() -> None:
    resultado = agregar_caracteristicas_basicas(
        _transaccion(
            latitud_cliente=-34.6037,
            longitud_cliente=-58.3816,
            latitud_comercio=-34.9011,
            longitud_comercio=-56.1645,
        )
    )

    # Distancia real ~205 km; alcanza con verificar que esté en un rango razonable.
    distancia = resultado.iloc[0]["distancia_cliente_comercio_km"]
    assert 190 < distancia < 220


def prueba_no_modifica_el_dataframe_original() -> None:
    original = _transaccion()
    columnas_originales = list(original.columns)

    agregar_caracteristicas_basicas(original)

    assert list(original.columns) == columnas_originales
