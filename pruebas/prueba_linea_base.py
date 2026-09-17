import numpy as np

from fraude.entrenamiento.costo import costo_total
from fraude.entrenamiento.linea_base import elegir_umbral_de_monto


def prueba_elige_un_umbral_que_separa_fraude_de_legitimas() -> None:
    rng = np.random.default_rng(0)
    legitimas = rng.uniform(1, 50, size=200)
    fraudes = rng.uniform(500, 2000, size=20)
    monto = np.concatenate([legitimas, fraudes])
    es_fraude = np.concatenate([np.zeros(200, dtype=bool), np.ones(20, dtype=bool)])

    resultado = elegir_umbral_de_monto(es_fraude, monto, costo_revision=10.0)

    prediccion = monto >= resultado.umbral
    assert costo_total(es_fraude, prediccion, monto, costo_revision=10.0) == resultado.costo
    # El umbral encontrado tiene que quedar entre el monto legitimo mas caro y el
    # fraude mas barato para separarlos correctamente.
    assert legitimas.max() < resultado.umbral <= fraudes.min()
