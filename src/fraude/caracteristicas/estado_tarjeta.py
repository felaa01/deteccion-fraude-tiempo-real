"""Estado por tarjeta y características históricas calculadas en el momento de la solicitud.

Es la contraparte online de `fraude.lotes.caracteristicas_historicas`. El almacén online no
puede guardar "cuántas transacciones tuvo la tarjeta en las últimas 24 horas" ya calculado,
porque ese número depende de la hora de la transacción que se está autorizando, y esa hora
todavía no existe cuando se actualiza el almacén. Lo que se guarda es el **estado crudo** de
la tarjeta (`EstadoTarjeta`) y la aritmética de ventanas se hace al recibir el pedido
(`calcular_caracteristicas`).

Las dos funciones son puras y no dependen de Spark, Redis ni Feast: las usan tanto el job de
streaming (`actualizar_estado`) como el servicio (`calcular_caracteristicas`), y la prueba de
training-serving skew las compara contra el cálculo batch.

Las reglas replican las del batch, que son las que definen qué vio "el pasado" de cada fila:
- Las ventanas son de transacciones **estrictamente anteriores**: dos transacciones de la
  misma tarjeta en el mismo segundo no se ven entre sí.
- Si la tarjeta no tiene historia, el acumulado y el ratio son nulos (no cero).
- "Categoría nueva" sí cuenta a las transacciones del mismo segundo que ya fueron aplicadas,
  porque el batch desempata por `id_transaccion` con `row_number`.
- La **transacción anterior** (para la distancia y la velocidad) es la previa de la tarjeta en
  el orden `(marca, id)`, como un `lag`: ahí sí cuenta una del mismo segundo. Así el estado solo
  necesita guardar la ubicación de la última transacción. Si el intervalo es de 0 segundos la
  velocidad es nula (no se puede dividir por cero), aunque la distancia sí se calcula.
"""

from dataclasses import dataclass
from datetime import UTC, datetime

from fraude.caracteristicas.geografia import distancia_haversine_km

SEGUNDOS_10_MINUTOS = 10 * 60
SEGUNDOS_1_HORA = 60 * 60
SEGUNDOS_24_HORAS = 24 * 60 * 60


def a_marca_unix(fecha: datetime) -> int:
    """Segundos desde la época; una fecha sin zona horaria se interpreta como UTC.

    Es la misma convención que usa el batch al fijar la zona horaria de la sesión de Spark
    en UTC (ver el bug de `to_timestamp` documentado en la semana 3).
    """
    if fecha.tzinfo is None:
        fecha = fecha.replace(tzinfo=UTC)
    return int(fecha.timestamp())


@dataclass(frozen=True)
class TransaccionTarjeta:
    """Lo mínimo de una transacción que hace falta para actualizar el estado."""

    id_transaccion: str
    marca_tiempo: int
    monto: float
    categoria: str
    latitud_comercio: float
    longitud_comercio: float


@dataclass(frozen=True)
class EstadoTarjeta:
    """Resumen de todo lo que se sabe de una tarjeta hasta su última transacción aplicada.

    `marcas_recientes` y `montos_recientes` son listas paralelas con las transacciones de
    las últimas 24 horas (la ventana más larga); el resto de la historia solo sobrevive
    como `cantidad_total` y `monto_total`. Así el tamaño del estado por tarjeta está
    acotado por la actividad de un día y no crece con la historia.

    `categorias_vistas` está siempre ordenada alfabéticamente: es la forma canónica, para que
    el estado calculado por lotes (arranque en frío) y el calculado transacción a transacción
    (streaming) se puedan comparar por igualdad sin depender del orden de aparición.

    `ultima_latitud_comercio` y `ultima_longitud_comercio` son la ubicación de la última
    transacción aplicada, para la distancia y la velocidad respecto de la siguiente.
    """

    cantidad_total: int
    monto_total: float
    categorias_vistas: tuple[str, ...]
    marcas_recientes: tuple[int, ...]
    montos_recientes: tuple[float, ...]
    ultima_marca: int
    ultimo_id_transaccion: str
    ultima_latitud_comercio: float
    ultima_longitud_comercio: float


@dataclass(frozen=True)
class CaracteristicasHistoricas:
    """Mismos nombres y semántica que las columnas que calcula el batch."""

    cantidad_transacciones_10min: int
    cantidad_transacciones_1h: int
    cantidad_transacciones_24h: int
    monto_acumulado_tarjeta: float | None
    ratio_monto_promedio_tarjeta: float | None
    categoria_nueva_para_tarjeta: bool
    distancia_transaccion_anterior_km: float | None
    velocidad_implicita_kmh: float | None


def actualizar_estado(
    estado: EstadoTarjeta | None, transaccion: TransaccionTarjeta
) -> EstadoTarjeta:
    """Aplica una transacción al estado de su tarjeta. Las transacciones llegan en orden.

    Es **idempotente**: Structured Streaming entrega cada lote al menos una vez, así que
    tras una falla puede reintentar un lote ya aplicado. Si el par (marca, id) no es
    posterior al de la última transacción aplicada, se devuelve el estado sin cambios. Esto
    supone orden por tarjeta, que Kafka garantiza dentro de una partición (la clave del
    mensaje es la tarjeta) y el productor respeta.
    """
    marca = transaccion.marca_tiempo
    if estado is None:
        return EstadoTarjeta(
            cantidad_total=1,
            monto_total=transaccion.monto,
            categorias_vistas=(transaccion.categoria,),
            marcas_recientes=(marca,),
            montos_recientes=(transaccion.monto,),
            ultima_marca=marca,
            ultimo_id_transaccion=transaccion.id_transaccion,
            ultima_latitud_comercio=transaccion.latitud_comercio,
            ultima_longitud_comercio=transaccion.longitud_comercio,
        )

    if (marca, transaccion.id_transaccion) <= (estado.ultima_marca, estado.ultimo_id_transaccion):
        return estado

    # Una transacción anterior a `marca - 24 h` no entra en la ventana de ninguna
    # transacción de `marca` en adelante, así que se puede descartar.
    corte = marca - SEGUNDOS_24_HORAS
    conservadas = [
        (m, monto)
        for m, monto in zip(estado.marcas_recientes, estado.montos_recientes, strict=True)
        if m >= corte
    ]
    categorias = estado.categorias_vistas
    if transaccion.categoria not in categorias:
        categorias = tuple(sorted((*categorias, transaccion.categoria)))
    return EstadoTarjeta(
        cantidad_total=estado.cantidad_total + 1,
        monto_total=estado.monto_total + transaccion.monto,
        categorias_vistas=categorias,
        marcas_recientes=(*(m for m, _ in conservadas), marca),
        montos_recientes=(*(monto for _, monto in conservadas), transaccion.monto),
        ultima_marca=marca,
        ultimo_id_transaccion=transaccion.id_transaccion,
        ultima_latitud_comercio=transaccion.latitud_comercio,
        ultima_longitud_comercio=transaccion.longitud_comercio,
    )


def calcular_caracteristicas(
    estado: EstadoTarjeta | None,
    marca_tiempo: int,
    monto: float,
    categoria: str,
    latitud_comercio: float,
    longitud_comercio: float,
) -> CaracteristicasHistoricas:
    """Características de una transacción que ocurre en `marca_tiempo`, dado el estado previo.

    `estado` es el de la tarjeta *antes* de esta transacción (`None` si no tiene historia).
    Supone `marca_tiempo >= estado.ultima_marca`: en producción la solicitud llega después
    de todo lo ya publicado; si no se cumple, el estado ya "vio el futuro" y se rechaza en
    lugar de devolver un número silenciosamente equivocado.
    """
    if estado is None:
        return CaracteristicasHistoricas(
            cantidad_transacciones_10min=0,
            cantidad_transacciones_1h=0,
            cantidad_transacciones_24h=0,
            monto_acumulado_tarjeta=None,
            ratio_monto_promedio_tarjeta=None,
            categoria_nueva_para_tarjeta=True,
            distancia_transaccion_anterior_km=None,
            velocidad_implicita_kmh=None,
        )
    if marca_tiempo < estado.ultima_marca:
        raise ValueError(
            "La transacción es anterior a la última aplicada al estado de la tarjeta "
            f"({marca_tiempo} < {estado.ultima_marca}): el estado ya incluye el futuro"
        )

    def contar(segundos: int) -> int:
        return sum(
            1 for m in estado.marcas_recientes if marca_tiempo - segundos <= m < marca_tiempo
        )

    # Las transacciones del mismo segundo están en el estado pero no cuentan como "previas".
    mismo_segundo = [
        monto_previo
        for m, monto_previo in zip(estado.marcas_recientes, estado.montos_recientes, strict=True)
        if m == marca_tiempo
    ]
    cantidad_previa = estado.cantidad_total - len(mismo_segundo)
    monto_previo = estado.monto_total - sum(mismo_segundo)

    # Transacción anterior = la última del estado (orden `(marca, id)`), aunque sea del mismo
    # segundo. Con 0 segundos de diferencia no hay velocidad, pero la distancia sí existe.
    distancia = distancia_haversine_km(
        estado.ultima_latitud_comercio,
        estado.ultima_longitud_comercio,
        latitud_comercio,
        longitud_comercio,
    )
    segundos = marca_tiempo - estado.ultima_marca
    velocidad = distancia / (segundos / 3600) if segundos > 0 else None

    tiene_historia = cantidad_previa > 0
    return CaracteristicasHistoricas(
        cantidad_transacciones_10min=contar(SEGUNDOS_10_MINUTOS),
        cantidad_transacciones_1h=contar(SEGUNDOS_1_HORA),
        cantidad_transacciones_24h=contar(SEGUNDOS_24_HORAS),
        monto_acumulado_tarjeta=monto_previo if tiene_historia else None,
        ratio_monto_promedio_tarjeta=(
            monto / (monto_previo / cantidad_previa) if tiene_historia else None
        ),
        categoria_nueva_para_tarjeta=categoria not in estado.categorias_vistas,
        distancia_transaccion_anterior_km=distancia,
        velocidad_implicita_kmh=velocidad,
    )
