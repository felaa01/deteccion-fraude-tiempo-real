import pandas as pd

from fraude.entrenamiento.division import dividir_temporalmente


def _transacciones(fechas: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fecha_hora_transaccion": pd.to_datetime(fechas),
            "monto": range(len(fechas)),
        }
    )


def prueba_corta_entrenamiento_y_validacion_por_fecha() -> None:
    crudo_entrenamiento = _transacciones(
        ["2019-10-31 23:59:00", "2019-11-01 00:00:00", "2020-01-01 00:00:00"]
    )
    crudo_prueba = _transacciones(["2020-07-01 00:00:00"])

    conjuntos = dividir_temporalmente(crudo_entrenamiento, crudo_prueba)

    assert len(conjuntos.entrenamiento) == 1
    assert len(conjuntos.validacion) == 2
    assert conjuntos.entrenamiento.iloc[0]["monto"] == 0


def prueba_prueba_final_queda_intacta() -> None:
    crudo_entrenamiento = _transacciones(["2019-06-01 00:00:00"])
    crudo_prueba = _transacciones(["2020-07-01 00:00:00", "2020-08-01 00:00:00"])

    conjuntos = dividir_temporalmente(crudo_entrenamiento, crudo_prueba)

    pd.testing.assert_frame_equal(conjuntos.prueba, crudo_prueba)


def prueba_respeta_un_corte_de_validacion_distinto() -> None:
    crudo_entrenamiento = _transacciones(["2019-03-01 00:00:00", "2019-05-01 00:00:00"])
    crudo_prueba = _transacciones(["2020-07-01 00:00:00"])

    conjuntos = dividir_temporalmente(
        crudo_entrenamiento, crudo_prueba, corte_validacion=pd.Timestamp("2019-04-01")
    )

    assert len(conjuntos.entrenamiento) == 1
    assert len(conjuntos.validacion) == 1
