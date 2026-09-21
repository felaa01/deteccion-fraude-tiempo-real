"""Características históricas por tarjeta, calculadas con PySpark sobre todo el histórico.

Point-in-time: cada fila usa exclusivamente transacciones estrictamente anteriores de la
misma tarjeta (nunca la transacción actual ni transacciones futuras de esa tarjeta). No es
un detalle menor de implementación: el valor que queda guardado para la transacción T es
lo que un servicio en producción podría haber sabido justo antes de T. Gracias a eso, el
join point-in-time de Feast (que matchea la fila con el timestamp más reciente
`<= tiempo de la entidad`) cae naturalmente en la propia fila de T sin necesitar un
desfasaje adicional en el join.

Dos transacciones de la misma tarjeta en el mismo segundo (mismo `marca_tiempo`) se
tratan como simultáneas: ninguna ve a la otra, ya que el corte de las ventanas de tiempo es
estrictamente anterior al segundo actual.
"""

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Las ventanas se definen una sola vez, en la ruta online, para que no se puedan desalinear.
from fraude.caracteristicas.estado_tarjeta import (
    SEGUNDOS_1_HORA,
    SEGUNDOS_10_MINUTOS,
    SEGUNDOS_24_HORAS,
)
from fraude.caracteristicas.geografia import RADIO_TIERRA_KM


def _distancia_haversine_km(
    latitud_1: Column, longitud_1: Column, latitud_2: Column, longitud_2: Column
) -> Column:
    """La misma fórmula que `fraude.caracteristicas.geografia`, como expresión de Spark."""
    fi_1, fi_2 = F.radians(latitud_1), F.radians(latitud_2)
    delta_fi = F.radians(latitud_2 - latitud_1)
    delta_lambda = F.radians(longitud_2 - longitud_1)
    a = F.sin(delta_fi / 2) ** 2 + F.cos(fi_1) * F.cos(fi_2) * F.sin(delta_lambda / 2) ** 2
    return 2 * RADIO_TIERRA_KM * F.asin(F.sqrt(a))


def agregar_caracteristicas_historicas(transacciones: DataFrame) -> DataFrame:
    """Devuelve `transacciones` con las características históricas por tarjeta agregadas.

    Requiere las columnas `numero_tarjeta`, `marca_tiempo` (segundos, derivados de la fecha
    de la transacción, no de `unix_time`), `id_transaccion`,
    `categoria`, `monto`, `latitud_comercio` y `longitud_comercio`. `id_transaccion` solo se
    usa para desempatar de forma determinística transacciones de una misma tarjeta con el
    mismo `marca_tiempo`.

    Agrega:
    - `cantidad_transacciones_10min` / `_1h` / `_24h`: transacciones previas de la tarjeta
      en cada ventana.
    - `monto_acumulado_tarjeta`: suma de los montos de todas las transacciones previas.
    - `ratio_monto_promedio_tarjeta`: monto actual sobre el promedio histórico de la
      tarjeta (nulo si es la primera transacción de la tarjeta, todavía sin historia).
    - `categoria_nueva_para_tarjeta`: si es la primera vez que esa tarjeta compra en esa
      categoría.
    - `distancia_transaccion_anterior_km` y `velocidad_implicita_kmh`: respecto de la
      transacción anterior de la tarjeta en el orden `(marca, id)` (un `lag`, que sí cuenta una
      del mismo segundo). Nulas en la primera transacción; la velocidad también es nula si el
      intervalo es de 0 segundos.
    """
    orden_por_tiempo = Window.partitionBy("numero_tarjeta").orderBy(
        F.col("marca_tiempo").cast("long")
    )
    ventana_10_min = orden_por_tiempo.rangeBetween(-SEGUNDOS_10_MINUTOS, -1)
    ventana_1_hora = orden_por_tiempo.rangeBetween(-SEGUNDOS_1_HORA, -1)
    ventana_24_horas = orden_por_tiempo.rangeBetween(-SEGUNDOS_24_HORAS, -1)
    ventana_historia_completa = orden_por_tiempo.rangeBetween(Window.unboundedPreceding, -1)
    orden_categoria_tarjeta = Window.partitionBy("numero_tarjeta", "categoria").orderBy(
        "marca_tiempo", "id_transaccion"
    )

    orden_transaccion_anterior = Window.partitionBy("numero_tarjeta").orderBy(
        "marca_tiempo", "id_transaccion"
    )
    distancia_anterior = _distancia_haversine_km(
        F.lag("latitud_comercio").over(orden_transaccion_anterior),
        F.lag("longitud_comercio").over(orden_transaccion_anterior),
        F.col("latitud_comercio"),
        F.col("longitud_comercio"),
    )
    segundos_desde_anterior = F.col("marca_tiempo") - F.lag("marca_tiempo").over(
        orden_transaccion_anterior
    )

    promedio_previo = F.avg("monto").over(ventana_historia_completa)

    return (
        transacciones.withColumn("cantidad_transacciones_10min", F.count("*").over(ventana_10_min))
        .withColumn("cantidad_transacciones_1h", F.count("*").over(ventana_1_hora))
        .withColumn("cantidad_transacciones_24h", F.count("*").over(ventana_24_horas))
        .withColumn("monto_acumulado_tarjeta", F.sum("monto").over(ventana_historia_completa))
        .withColumn("ratio_monto_promedio_tarjeta", F.col("monto") / promedio_previo)
        .withColumn(
            "categoria_nueva_para_tarjeta",
            F.row_number().over(orden_categoria_tarjeta) == 1,
        )
        .withColumn("distancia_transaccion_anterior_km", distancia_anterior)
        .withColumn(
            "velocidad_implicita_kmh",
            # Sin `otherwise`: con intervalo de 0 s (o sin anterior) queda nula.
            F.when(
                segundos_desde_anterior > 0,
                F.col("distancia_transaccion_anterior_km") / (segundos_desde_anterior / 3600),
            ),
        )
    )
