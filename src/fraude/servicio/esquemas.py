"""Esquemas de entrada y salida de la API de predicción."""

from datetime import datetime

from pydantic import BaseModel, Field


class TransaccionEntrada(BaseModel):
    """Los campos crudos de una transacción, necesarios para calcular sus características.

    Son los mismos campos que usa `agregar_caracteristicas_basicas` en el entrenamiento
    (`fraude.entrenamiento.caracteristicas_basicas`) más las columnas que el modelo usa
    directamente, para no sufrir de training-serving skew (regla 3): la característica se
    calcula una sola vez, con la misma función, offline y online.
    """

    fecha_hora_transaccion: datetime
    monto: float = Field(gt=0)
    categoria: str
    genero: str
    fecha_nacimiento: datetime
    latitud_cliente: float
    longitud_cliente: float
    latitud_comercio: float
    longitud_comercio: float
    poblacion_ciudad: int = Field(gt=0)


class RespuestaPrediccion(BaseModel):
    """Resultado de puntuar una transacción con el campeón vigente."""

    probabilidad_fraude: float
    es_fraude: bool
    umbral_usado: float
    version_modelo: str
