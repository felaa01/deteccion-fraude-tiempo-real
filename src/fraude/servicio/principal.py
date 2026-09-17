"""API de predicción de fraude: puntúa transacciones con el campeón vigente de MLflow."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

import numpy as np
import pandas as pd
from fastapi import FastAPI

from fraude.entrenamiento.caracteristicas_basicas import agregar_caracteristicas_basicas
from fraude.servicio.esquemas import RespuestaPrediccion, TransaccionEntrada
from fraude.servicio.modelo_actual import Campeon, cargar_campeon


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI) -> AsyncIterator[None]:
    app.state.campeon = cargar_campeon()
    yield


app = FastAPI(title="Detección de fraude en tiempo real", lifespan=ciclo_de_vida)


@app.get("/salud")
def salud() -> dict[str, str]:
    return {"estado": "ok"}


@app.post("/predecir")
def predecir(transaccion: TransaccionEntrada) -> RespuestaPrediccion:
    campeon = cast(Campeon, app.state.campeon)

    fila = pd.DataFrame([transaccion.model_dump()])
    fila_con_caracteristicas = agregar_caracteristicas_basicas(fila)
    probabilidad = float(np.asarray(campeon.modelo.predict(fila_con_caracteristicas))[0])

    return RespuestaPrediccion(
        probabilidad_fraude=probabilidad,
        es_fraude=probabilidad >= campeon.umbral,
        umbral_usado=campeon.umbral,
        version_modelo=campeon.version,
    )
