"""Características históricas por tarjeta, calculadas con PySpark sobre todo el histórico.

Point-in-time: cada fila usa exclusivamente transacciones estrictamente anteriores de la
misma tarjeta (nunca la transacción actual ni transacciones futuras de esa tarjeta). No es
un detalle menor de implementación: el valor que queda guardado para la transacción T es
lo que un servicio en producción podría haber sabido justo antes de T. Gracias a eso, el
join point-in-time de Feast (que matchea la fila con el timestamp más reciente
`<= tiempo de la entidad`) cae naturalmente en la propia fila de T sin necesitar un
desfasaje adicional en el join.

Dos transacciones de la misma tarjeta en el mismo segundo (mismo `marca_tiempo_unix`) se
tratan como simultáneas: ninguna ve a la otra, ya que el corte de las ventanas de tiempo es
estrictamente anterior al segundo actual.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

SEGUNDOS_10_MINUTOS = 10 * 60
SEGUNDOS_1_HORA = 60 * 60
SEGUNDOS_24_HORAS = 24 * 60 * 60


def agregar_caracteristicas_historicas(transacciones: DataFrame) -> DataFrame:
    """Devuelve `transacciones` con las características históricas por tarjeta agregadas.

    Requiere las columnas `numero_tarjeta`, `marca_tiempo_unix`, `id_transaccion`,
    `categoria` y `monto`. `id_transaccion` solo se usa para desempatar de forma
    determinística transacciones de una misma tarjeta con el mismo `marca_tiempo_unix`.

    Agrega:
    - `cantidad_transacciones_10min` / `_1h` / `_24h`: transacciones previas de la tarjeta
      en cada ventana.
    - `monto_acumulado_tarjeta`: suma de los montos de todas las transacciones previas.
    - `ratio_monto_promedio_tarjeta`: monto actual sobre el promedio histórico de la
      tarjeta (nulo si es la primera transacción de la tarjeta, todavía sin historia).
    - `categoria_nueva_para_tarjeta`: si es la primera vez que esa tarjeta compra en esa
      categoría.
    """
    orden_por_tiempo = Window.partitionBy("numero_tarjeta").orderBy(
        F.col("marca_tiempo_unix").cast("long")
    )
    ventana_10_min = orden_por_tiempo.rangeBetween(-SEGUNDOS_10_MINUTOS, -1)
    ventana_1_hora = orden_por_tiempo.rangeBetween(-SEGUNDOS_1_HORA, -1)
    ventana_24_horas = orden_por_tiempo.rangeBetween(-SEGUNDOS_24_HORAS, -1)
    ventana_historia_completa = orden_por_tiempo.rangeBetween(Window.unboundedPreceding, -1)
    orden_categoria_tarjeta = Window.partitionBy("numero_tarjeta", "categoria").orderBy(
        "marca_tiempo_unix", "id_transaccion"
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
    )
