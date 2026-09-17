import numpy as np
import pandas as pd

from fraude.entrenamiento.modelo import entrenar, predecir_probabilidad


def _transacciones_sinteticas(cantidad: int, semilla: int) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    es_fraude = rng.random(cantidad) < 0.1
    monto_fraude = rng.uniform(500, 2000, cantidad)
    monto_legitimo = rng.uniform(1, 200, cantidad)
    return pd.DataFrame(
        {
            "monto": np.where(es_fraude, monto_fraude, monto_legitimo),
            "hora_del_dia": rng.integers(0, 24, cantidad),
            "dia_de_la_semana": rng.integers(0, 7, cantidad),
            "edad_titular": rng.uniform(18, 90, cantidad),
            "distancia_cliente_comercio_km": rng.uniform(0, 500, cantidad),
            "poblacion_ciudad": rng.integers(100, 1_000_000, cantidad),
            "categoria": rng.choice(["shopping_net", "grocery_pos", "gas_transport"], cantidad),
            "genero": rng.choice(["F", "M"], cantidad),
            "es_fraude": es_fraude.astype(int),
        }
    )


def prueba_predice_probabilidades_entre_cero_y_uno() -> None:
    entrenamiento = _transacciones_sinteticas(300, semilla=0)
    modelo = entrenar(entrenamiento, semilla=0)

    probabilidades = predecir_probabilidad(modelo, entrenamiento)

    assert probabilidades.shape == (300,)
    assert (probabilidades >= 0).all()
    assert (probabilidades <= 1).all()


def prueba_asigna_mayor_probabilidad_promedio_al_fraude() -> None:
    entrenamiento = _transacciones_sinteticas(400, semilla=1)
    modelo = entrenar(entrenamiento, semilla=1)

    probabilidades = predecir_probabilidad(modelo, entrenamiento)
    es_fraude = entrenamiento["es_fraude"].astype(bool).to_numpy()

    assert probabilidades[es_fraude].mean() > probabilidades[~es_fraude].mean()
