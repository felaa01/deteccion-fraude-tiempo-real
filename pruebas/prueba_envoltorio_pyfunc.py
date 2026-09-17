import json
from pathlib import Path

import joblib
import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch

from fraude.entrenamiento.envoltorio_pyfunc import EnvoltorioLightGBM, EnvoltorioPyTorch
from fraude.entrenamiento.modelo import entrenar
from fraude.entrenamiento.modelo_pytorch import entrenar_red, preprocesador_a_dict


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


def prueba_envoltorio_lightgbm_predice_igual_que_el_modelo_directo(tmp_path: Path) -> None:
    entrenamiento = _transacciones_sinteticas(300, semilla=0)
    modelo = entrenar(entrenamiento, semilla=0)

    ruta_modelo = tmp_path / "modelo.joblib"
    joblib.dump(modelo, ruta_modelo)
    ruta_pyfunc = tmp_path / "pyfunc"
    mlflow.pyfunc.save_model(
        path=str(ruta_pyfunc),
        python_model=EnvoltorioLightGBM(),
        artifacts={"modelo": str(ruta_modelo)},
    )

    modelo_cargado = mlflow.pyfunc.load_model(str(ruta_pyfunc))
    probabilidades = modelo_cargado.predict(entrenamiento)

    assert probabilidades.shape == (300,)
    assert (probabilidades >= 0).all()
    assert (probabilidades <= 1).all()


def prueba_envoltorio_pytorch_predice_igual_que_el_modelo_directo(tmp_path: Path) -> None:
    entrenamiento = _transacciones_sinteticas(300, semilla=1)
    validacion = _transacciones_sinteticas(100, semilla=2)
    modelo, preprocesador = entrenar_red(
        entrenamiento,
        validacion,
        monto_validacion=validacion["monto"].to_numpy(),
        costo_revision=10.0,
        epocas=2,
    )

    ruta_modelo = tmp_path / "modelo.pt"
    ruta_preprocesador = tmp_path / "preprocesador.json"
    torch.save(modelo, ruta_modelo)
    ruta_preprocesador.write_text(json.dumps(preprocesador_a_dict(preprocesador)))
    ruta_pyfunc = tmp_path / "pyfunc"
    mlflow.pyfunc.save_model(
        path=str(ruta_pyfunc),
        python_model=EnvoltorioPyTorch(),
        artifacts={"modelo": str(ruta_modelo), "preprocesador": str(ruta_preprocesador)},
    )

    modelo_cargado = mlflow.pyfunc.load_model(str(ruta_pyfunc))
    probabilidades = modelo_cargado.predict(entrenamiento)

    assert probabilidades.shape == (300,)
    assert (probabilidades >= 0).all()
    assert (probabilidades <= 1).all()
