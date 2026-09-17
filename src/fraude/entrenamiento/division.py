"""División temporal de las transacciones en entrenamiento, validación y prueba.

Regla no negociable del proyecto: solo partición temporal, nunca aleatoria. El dataset
de Kaggle ya trae `fraudTrain.csv` y `fraudTest.csv` separados cronológicamente; acá se
corta además `fraudTrain.csv` en entrenamiento/validación para poder elegir el umbral de
costo y comparar campeón contra retador sin tocar el conjunto de prueba final.
"""

from dataclasses import dataclass

import pandas as pd

# Corte por defecto: deja ~10 meses (2019-01 a 2019-10) para entrenar y ~8 meses
# (2019-11 a 2020-06, fin de fraudTrain.csv) para validación.
CORTE_VALIDACION_POR_DEFECTO = pd.Timestamp("2019-11-01")


@dataclass(frozen=True)
class ConjuntosDivididos:
    """Los tres conjuntos resultantes de la división temporal."""

    entrenamiento: pd.DataFrame
    validacion: pd.DataFrame
    prueba: pd.DataFrame


def dividir_temporalmente(
    transacciones_entrenamiento: pd.DataFrame,
    transacciones_prueba: pd.DataFrame,
    corte_validacion: pd.Timestamp = CORTE_VALIDACION_POR_DEFECTO,
) -> ConjuntosDivididos:
    """Divide las transacciones en entrenamiento/validación/prueba por fecha.

    `transacciones_entrenamiento` (fraudTrain.csv ya cargado) se corta en
    `corte_validacion`: lo anterior queda como entrenamiento, el resto como
    validación. `transacciones_prueba` (fraudTest.csv ya cargado) se usa completo
    y sin modificar como conjunto de prueba final.
    """
    columna_fecha = transacciones_entrenamiento["fecha_hora_transaccion"]
    return ConjuntosDivididos(
        entrenamiento=transacciones_entrenamiento[columna_fecha < corte_validacion],
        validacion=transacciones_entrenamiento[columna_fecha >= corte_validacion],
        prueba=transacciones_prueba,
    )
