from pathlib import Path

import pandas as pd
import pytest
from feast import Project
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

from fraude.caracteristicas.almacen_estado import leer_estados
from fraude.caracteristicas.arranque_en_frio import materializar_estado_inicial
from fraude.caracteristicas.estado_tarjeta import (
    EstadoTarjeta,
    TransaccionTarjeta,
    actualizar_estado,
)
from fraude.caracteristicas.tienda import crear_tienda_online
from fraude.lotes.estado_inicial import calcular_estado_tarjeta

ESQUEMA = StructType(
    [
        StructField("id_transaccion", StringType()),
        StructField("numero_tarjeta", LongType()),
        StructField("marca_tiempo_unix", LongType()),
        StructField("categoria", StringType()),
        StructField("monto", DoubleType()),
    ]
)

CONEXION_REDIS_DE_PRUEBA = "localhost:6379,db=1"
PROYECTO_DE_PRUEBA = "prueba_estado_inicial"


def _con_fecha(sesion_spark: SparkSession, transacciones: pd.DataFrame) -> DataFrame:
    return sesion_spark.createDataFrame(transacciones, ESQUEMA).withColumn(
        "fecha_hora_transaccion", col("marca_tiempo_unix").cast("timestamp")
    )


def _estado_secuencial(transacciones: pd.DataFrame) -> dict[int, EstadoTarjeta]:
    """Lo que daría el streaming: aplicar `actualizar_estado` una transacción por vez."""
    estados: dict[int, EstadoTarjeta] = {}
    orden = transacciones.sort_values(["marca_tiempo_unix", "id_transaccion"])
    for fila in orden.to_dict("records"):
        tarjeta = fila["numero_tarjeta"]
        estados[tarjeta] = actualizar_estado(
            estados.get(tarjeta),
            TransaccionTarjeta(
                fila["id_transaccion"], fila["marca_tiempo_unix"], fila["monto"], fila["categoria"]
            ),
        )
    return estados


def _assert_equivalentes(obtenido: EstadoTarjeta | None, esperado: EstadoTarjeta) -> None:
    """Igualdad exacta salvo los montos: Spark y Python suman en distinto orden."""
    assert obtenido is not None
    assert obtenido.cantidad_total == esperado.cantidad_total
    assert obtenido.categorias_vistas == esperado.categorias_vistas
    assert obtenido.marcas_recientes == esperado.marcas_recientes
    assert obtenido.ultima_marca == esperado.ultima_marca
    assert obtenido.ultimo_id_transaccion == esperado.ultimo_id_transaccion
    assert obtenido.monto_total == pytest.approx(esperado.monto_total, rel=1e-9)
    assert obtenido.montos_recientes == pytest.approx(esperado.montos_recientes, rel=1e-9)


def prueba_estado_inicial_de_spark_coincide_con_aplicar_actualizar_estado(
    sesion_spark: SparkSession, transacciones_aleatorias: pd.DataFrame
) -> None:
    """El lote (agregaciones de Spark) y el streaming (fila a fila) dan el mismo estado."""
    lote = calcular_estado_tarjeta(_con_fecha(sesion_spark, transacciones_aleatorias)).collect()
    esperado = _estado_secuencial(transacciones_aleatorias)

    assert len(lote) == len(esperado) == 4
    for fila in lote:
        obtenido = EstadoTarjeta(
            cantidad_total=fila["cantidad_total"],
            monto_total=fila["monto_total"],
            categorias_vistas=tuple(fila["categorias_vistas"]),
            marcas_recientes=tuple(fila["marcas_recientes"]),
            montos_recientes=tuple(fila["montos_recientes"]),
            ultima_marca=fila["ultima_marca"],
            ultimo_id_transaccion=fila["ultimo_id_transaccion"],
        )
        _assert_equivalentes(obtenido, esperado[fila["numero_tarjeta"]])


@pytest.mark.integracion
def prueba_arranque_en_frio_deja_en_redis_el_mismo_estado(
    sesion_spark: SparkSession,
    transacciones_aleatorias: pd.DataFrame,
    tmp_path: Path,
    redis_de_prueba: None,
) -> None:
    """Spark -> Parquet -> `materialize` -> Redis -> lectura, contra el estado secuencial."""
    ruta = tmp_path / "estado_inicial.parquet"
    calcular_estado_tarjeta(_con_fecha(sesion_spark, transacciones_aleatorias)).coalesce(
        1
    ).write.parquet(str(ruta))
    tienda = crear_tienda_online(
        tmp_path / "registro.db", CONEXION_REDIS_DE_PRUEBA, proyecto=PROYECTO_DE_PRUEBA
    )

    materializar_estado_inicial(tienda, ruta, Project(name=PROYECTO_DE_PRUEBA))
    esperado = _estado_secuencial(transacciones_aleatorias)
    leidos = leer_estados(tienda, list(esperado))

    for tarjeta, estado in esperado.items():
        _assert_equivalentes(leidos[tarjeta], estado)
