"""Wrapper pyfunc de MLflow para servir cualquier arquitectura (LightGBM o PyTorch) igual.

Sin este wrapper, el servicio de FastAPI necesitaría saber qué flavor tiene la versión con
el alias `campeon` para elegir cómo cargarla y cómo pedirle la probabilidad de fraude
(LightGBM necesita `predict_proba`, PyTorch necesita el preprocesador + sigmoide). Envolviendo
las dos arquitecturas en la misma interfaz pyfunc, el servicio siempre hace
`mlflow.pyfunc.load_model(...).predict(df)` y recibe la probabilidad de fraude, sin importar
qué hay detrás del alias.

El `model_input` que reciben ambos wrappers ya tiene las características calculadas (el mismo
`agregar_caracteristicas_basicas` que se usa en el entrenamiento) — el wrapper solo envuelve el
modelo, no vuelve a calcular características.
"""

import json
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from mlflow.pyfunc.model import PythonModel, PythonModelContext

from fraude.entrenamiento.modelo import predecir_probabilidad as predecir_probabilidad_lightgbm
from fraude.entrenamiento.modelo_pytorch import Preprocesador
from fraude.entrenamiento.modelo_pytorch import predecir_probabilidad as predecir_probabilidad_red


class EnvoltorioLightGBM(PythonModel):
    """Envuelve un LGBMClassifier para que devuelva P(fraude) vía la interfaz pyfunc."""

    def load_context(self, context: PythonModelContext) -> None:
        self.modelo = joblib.load(context.artifacts["modelo"])

    def predict(
        self,
        context: PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> npt.NDArray[np.float64]:
        return predecir_probabilidad_lightgbm(self.modelo, model_input)


class EnvoltorioPyTorch(PythonModel):
    """Envuelve la red de PyTorch y su preprocesador para devolver P(fraude) vía pyfunc."""

    def load_context(self, context: PythonModelContext) -> None:
        self.modelo = torch.load(context.artifacts["modelo"], weights_only=False)
        self.modelo.eval()
        with open(context.artifacts["preprocesador"]) as archivo:
            datos = json.load(archivo)
        self.preprocesador = Preprocesador(
            medias=np.array(datos["medias"], dtype=np.float64),
            desvios=np.array(datos["desvios"], dtype=np.float64),
            vocabularios=datos["vocabularios"],
        )

    def predict(
        self,
        context: PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> npt.NDArray[np.float64]:
        return predecir_probabilidad_red(self.modelo, self.preprocesador, model_input)
