"""API de predicción de fraude: puntúa transacciones con el campeón vigente de MLflow."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import cast

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from feast import FeatureStore
from redis.exceptions import RedisError

from fraude.caracteristicas.almacen_estado import leer_estados
from fraude.caracteristicas.estado_tarjeta import a_marca_unix, calcular_caracteristicas
from fraude.entrenamiento.caracteristicas_basicas import agregar_caracteristicas_basicas
from fraude.servicio.esquemas import (
    CaracteristicasHistoricasRespuesta,
    RespuestaPrediccion,
    TransaccionEntrada,
)
from fraude.servicio.modelo_actual import Campeon, cargar_campeon
from fraude.servicio.tienda_estado import cargar_tienda


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI) -> AsyncIterator[None]:
    app.state.campeon = cargar_campeon()
    app.state.tienda = cargar_tienda()
    yield


app = FastAPI(title="Detección de fraude en tiempo real", lifespan=ciclo_de_vida)


@app.get("/salud")
def salud() -> dict[str, str]:
    return {"estado": "ok"}


@app.post("/predecir")
def predecir(transaccion: TransaccionEntrada) -> RespuestaPrediccion:
    campeon = cast(Campeon, app.state.campeon)
    tienda = cast(FeatureStore, app.state.tienda)

    # Sin Redis no hay forma de saber la historia de la tarjeta: se falla en vez de tratarla
    # como nueva, que devolvería características equivocadas en silencio.
    try:
        estado = leer_estados(tienda, [transaccion.numero_tarjeta])[transaccion.numero_tarjeta]
    except RedisError as error:
        raise HTTPException(
            status_code=503, detail="No se pudo leer el estado de la tarjeta del almacén online"
        ) from error

    try:
        historicas = calcular_caracteristicas(
            estado,
            a_marca_unix(transaccion.fecha_hora_transaccion),
            transaccion.monto,
            transaccion.categoria,
            transaccion.latitud_comercio,
            transaccion.longitud_comercio,
        )
    except ValueError as error:
        # La solicitud es anterior a lo que el estado ya incorporó: choca con el estado actual.
        raise HTTPException(status_code=409, detail=str(error)) from error

    # El modelo no conoce el número de tarjeta: es una clave de búsqueda, no una característica.
    fila = pd.DataFrame([transaccion.model_dump(exclude={"numero_tarjeta"})])
    fila_con_caracteristicas = agregar_caracteristicas_basicas(fila)
    probabilidad = float(np.asarray(campeon.modelo.predict(fila_con_caracteristicas))[0])

    return RespuestaPrediccion(
        probabilidad_fraude=probabilidad,
        es_fraude=probabilidad >= campeon.umbral,
        umbral_usado=campeon.umbral,
        version_modelo=campeon.version,
        tarjeta_con_historia=estado is not None,
        caracteristicas_historicas=CaracteristicasHistoricasRespuesta.model_validate(
            asdict(historicas)
        ),
    )
