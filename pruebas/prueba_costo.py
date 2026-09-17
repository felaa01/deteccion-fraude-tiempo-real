import numpy as np

from fraude.entrenamiento.costo import costo_total, elegir_umbral


def prueba_costo_verdadero_negativo_es_cero() -> None:
    es_fraude = np.array([False])
    prediccion = np.array([False])
    monto = np.array([100.0])

    assert costo_total(es_fraude, prediccion, monto) == 0.0


def prueba_costo_falso_positivo_es_el_costo_de_revision() -> None:
    es_fraude = np.array([False])
    prediccion = np.array([True])
    monto = np.array([100.0])

    assert costo_total(es_fraude, prediccion, monto, costo_revision=10.0) == 10.0


def prueba_costo_falso_negativo_es_el_monto_completo() -> None:
    es_fraude = np.array([True])
    prediccion = np.array([False])
    monto = np.array([250.0])

    assert costo_total(es_fraude, prediccion, monto) == 250.0


def prueba_costo_verdadero_positivo_es_el_costo_de_revision() -> None:
    es_fraude = np.array([True])
    prediccion = np.array([True])
    monto = np.array([250.0])

    assert costo_total(es_fraude, prediccion, monto, costo_revision=10.0) == 10.0


def prueba_elegir_umbral_no_marca_una_legitima_de_baja_probabilidad() -> None:
    # Fraude caro (2000, prob. 0.9) y legitima barata (5, prob. 0.2): marcar la
    # legitima solo suma el costo fijo de revision sin evitar ninguna perdida, asi
    # que el umbral optimo la deja pasar y solo atrapa el fraude.
    es_fraude = np.array([True, False])
    probabilidad = np.array([0.9, 0.2])
    monto = np.array([2000.0, 5.0])

    resultado = elegir_umbral(es_fraude, probabilidad, monto, costo_revision=10.0)

    assert 0.2 < resultado.umbral <= 0.9
    assert resultado.costo == 10.0


def prueba_elegir_umbral_no_marca_nada_si_no_hay_fraude() -> None:
    es_fraude = np.array([False, False])
    probabilidad = np.array([0.6, 0.7])
    monto = np.array([50.0, 80.0])

    resultado = elegir_umbral(es_fraude, probabilidad, monto, costo_revision=10.0)

    assert resultado.umbral > 0.7
    assert resultado.costo == 0.0
