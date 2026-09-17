"""Línea base de reglas: marca como fraude toda transacción que supere un monto.

Sirve de piso de comparación para el gradient boosting: si el modelo no reduce el
costo de negocio frente a esta regla simple, no se justifica la complejidad extra.
"""

import numpy as np
import numpy.typing as npt

from fraude.entrenamiento.costo import UmbralOptimo, elegir_umbral

CANTIDAD_DE_CANDIDATOS = 99


def elegir_umbral_de_monto(
    es_fraude: npt.NDArray[np.bool_],
    monto: npt.NDArray[np.float64],
    costo_revision: float = 10.0,
) -> UmbralOptimo:
    """Elige, sobre un conjunto de validación, el umbral de monto que minimiza el costo.

    La regla es "marcar como fraude toda transacción con `monto >= umbral`". Es el mismo
    problema de optimización que elegir un umbral de probabilidad: se reutiliza
    `elegir_umbral` pasando el propio monto como score a umbralizar.
    """
    candidatos = np.quantile(monto, np.linspace(0.01, 0.99, CANTIDAD_DE_CANDIDATOS))
    return elegir_umbral(
        es_fraude, monto, monto, candidatos=candidatos, costo_revision=costo_revision
    )
