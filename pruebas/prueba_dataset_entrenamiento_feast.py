from pathlib import Path

import pandas as pd
from feast import FeatureStore

from fraude.caracteristicas.dataset_entrenamiento import (
    agregar_caracteristicas_historicas_point_in_time,
    aplicar_definiciones,
)
from fraude.caracteristicas.tienda import crear_tienda

# Dos transacciones de la tarjeta 111, como las dejaría `fraude.lotes.calcular_historico`:
# cada fila ya trae sus propias características "sin fuga" (calculadas solo con
# transacciones estrictamente anteriores).
HISTORICO = pd.DataFrame(
    {
        "numero_tarjeta": [111, 111],
        "fecha_hora_transaccion": [
            pd.Timestamp("2019-01-01 10:00:00"),
            pd.Timestamp("2019-01-01 10:05:00"),
        ],
        "cantidad_transacciones_10min": pd.array([0, 1], dtype="Int64"),
        "cantidad_transacciones_1h": pd.array([0, 1], dtype="Int64"),
        "cantidad_transacciones_24h": pd.array([0, 1], dtype="Int64"),
        "monto_acumulado_tarjeta": [float("nan"), 10.0],
        "ratio_monto_promedio_tarjeta": [float("nan"), 2.0],
        "categoria_nueva_para_tarjeta": [True, False],
        "distancia_transaccion_anterior_km": [float("nan"), 111.19],
        "velocidad_implicita_kmh": [float("nan"), 1334.3],
    }
)


def _tienda_con_historico(tmp_path: Path) -> tuple[FeatureStore, Path]:
    ruta_historico = tmp_path / "historico.parquet"
    HISTORICO.to_parquet(ruta_historico)
    tienda = crear_tienda(tmp_path / "registro.db", tmp_path / "online.db")
    aplicar_definiciones(tienda, ruta_historico)
    return tienda, ruta_historico


def prueba_matchea_la_fila_con_el_mismo_timestamp_de_la_transaccion(
    tmp_path: Path,
) -> None:
    tienda, _ = _tienda_con_historico(tmp_path)
    transacciones = pd.DataFrame(
        {
            "numero_tarjeta": [111],
            "fecha_hora_transaccion": [pd.Timestamp("2019-01-01 10:05:00")],
            "monto": [20.0],
            "es_fraude": [0],
        }
    )

    resultado = agregar_caracteristicas_historicas_point_in_time(tienda, transacciones)

    fila = resultado.iloc[0]
    assert fila["cantidad_transacciones_10min"] == 1
    assert fila["monto_acumulado_tarjeta"] == 10.0
    assert fila["ratio_monto_promedio_tarjeta"] == 2.0
    # Las columnas propias de la transacción se preservan.
    assert fila["monto"] == 20.0
    assert fila["es_fraude"] == 0
    assert fila["fecha_hora_transaccion"] == pd.Timestamp("2019-01-01 10:05:00", tz="UTC")


def prueba_no_usa_una_fila_de_historico_posterior_a_la_transaccion(
    tmp_path: Path,
) -> None:
    tienda, _ = _tienda_con_historico(tmp_path)
    # Timestamp de la entidad apenas antes de la segunda fila del histórico: el join
    # point-in-time tiene que quedarse con la primera fila (10:00), no con la de 10:05.
    transacciones = pd.DataFrame(
        {
            "numero_tarjeta": [111],
            "fecha_hora_transaccion": [pd.Timestamp("2019-01-01 10:04:59")],
            "monto": [15.0],
        }
    )

    resultado = agregar_caracteristicas_historicas_point_in_time(tienda, transacciones)

    fila = resultado.iloc[0]
    assert fila["cantidad_transacciones_10min"] == 0
    assert pd.isna(fila["monto_acumulado_tarjeta"])
