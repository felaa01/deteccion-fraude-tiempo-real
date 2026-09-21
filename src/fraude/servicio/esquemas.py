"""Esquemas de entrada y salida de la API de predicción."""

from datetime import datetime

from pydantic import BaseModel, Field


class TransaccionEntrada(BaseModel):
    """Los campos crudos de una transacción, necesarios para calcular sus características.

    Son los mismos campos que usa `agregar_caracteristicas_basicas` en el entrenamiento
    (`fraude.entrenamiento.caracteristicas_basicas`) más las columnas que el modelo usa
    directamente, para no sufrir de training-serving skew (regla 3): la característica se
    calcula una sola vez, con la misma función, offline y online.

    `numero_tarjeta` no lo usa el modelo: sirve para buscar el estado de la tarjeta en el
    almacén online y calcular las características históricas.
    """

    numero_tarjeta: int
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


class CaracteristicasHistoricasRespuesta(BaseModel):
    """Las características de historia de la tarjeta, calculadas en el momento de la solicitud.

    Mismos nombres y semántica que `CaracteristicasHistoricas`, de
    `fraude.caracteristicas.estado_tarjeta`.
    """

    cantidad_transacciones_10min: int
    cantidad_transacciones_1h: int
    cantidad_transacciones_24h: int
    monto_acumulado_tarjeta: float | None
    ratio_monto_promedio_tarjeta: float | None
    categoria_nueva_para_tarjeta: bool
    distancia_transaccion_anterior_km: float | None
    velocidad_implicita_kmh: float | None


class RespuestaPrediccion(BaseModel):
    """Resultado de puntuar una transacción con el campeón vigente.

    Las características históricas se devuelven junto con la predicción, pero el campeón
    actual todavía no las usa (se entrenó solo con las básicas): sirven para validar el
    training-serving skew de punta a punta antes de entrenar un modelo que las consuma.
    """

    probabilidad_fraude: float
    es_fraude: bool
    umbral_usado: float
    version_modelo: str
    tarjeta_con_historia: bool
    caracteristicas_historicas: CaracteristicasHistoricasRespuesta
