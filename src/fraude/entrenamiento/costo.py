"""Métrica de costo de negocio y selección de umbral.

El accuracy no sirve para fraude (regla no negociable del proyecto): en su lugar se usa
una matriz de costo dependiente del monto. Un falso negativo cuesta el monto completo de
la transacción (se pierde la plata); un falso positivo o un verdadero positivo cuestan lo
mismo, un costo fijo de revisión/fricción con el cliente (igual hay que investigar la
transacción, se haya bloqueado a tiempo o no). Un verdadero negativo no cuesta nada.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

COSTO_REVISION_POR_DEFECTO = 10.0


def costo_total(
    es_fraude: npt.NDArray[np.bool_],
    prediccion_fraude: npt.NDArray[np.bool_],
    monto: npt.NDArray[np.float64],
    costo_revision: float = COSTO_REVISION_POR_DEFECTO,
) -> float:
    """Suma el costo de negocio de un conjunto de predicciones.

    - Verdadero negativo (no era fraude, no se marcó): costo 0.
    - Falso positivo (no era fraude, se marcó): `costo_revision`.
    - Verdadero positivo (era fraude, se marcó): `costo_revision` (se investiga igual).
    - Falso negativo (era fraude, no se marcó): el monto completo de la transacción.
    """
    costo_por_marcar = prediccion_fraude * costo_revision
    costo_por_no_detectar = (~prediccion_fraude) * es_fraude * monto
    return float(np.sum(costo_por_marcar + costo_por_no_detectar))


@dataclass(frozen=True)
class UmbralOptimo:
    """El umbral de probabilidad que minimiza el costo total, y ese costo."""

    umbral: float
    costo: float


def elegir_umbral(
    es_fraude: npt.NDArray[np.bool_],
    probabilidad_fraude: npt.NDArray[np.float64],
    monto: npt.NDArray[np.float64],
    candidatos: Iterable[float] = np.linspace(0.01, 0.99, 99),
    costo_revision: float = COSTO_REVISION_POR_DEFECTO,
) -> UmbralOptimo:
    """Barre umbrales candidatos y devuelve el que minimiza el costo total.

    Se evalúa sobre un conjunto de validación con etiquetas ya disponibles en ese
    momento (nunca sobre el conjunto de prueba final).
    """
    mejor = UmbralOptimo(umbral=0.5, costo=float("inf"))
    for umbral in candidatos:
        prediccion = probabilidad_fraude >= umbral
        costo = costo_total(es_fraude, prediccion, monto, costo_revision)
        if costo < mejor.costo:
            mejor = UmbralOptimo(umbral=float(umbral), costo=costo)
    return mejor
