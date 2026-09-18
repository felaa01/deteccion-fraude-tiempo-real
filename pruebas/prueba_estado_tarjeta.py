import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

from fraude.caracteristicas.estado_tarjeta import (
    SEGUNDOS_24_HORAS,
    EstadoTarjeta,
    TransaccionTarjeta,
    actualizar_estado,
    calcular_caracteristicas,
)
from fraude.lotes.caracteristicas_historicas import agregar_caracteristicas_historicas


def _tx(
    id_transaccion: str, marca: int, monto: float = 10.0, categoria: str = "comida"
) -> TransaccionTarjeta:
    return TransaccionTarjeta(id_transaccion, marca, monto, categoria)


def _aplicar(*transacciones: TransaccionTarjeta) -> EstadoTarjeta:
    estado: EstadoTarjeta | None = None
    for transaccion in transacciones:
        estado = actualizar_estado(estado, transaccion)
    assert estado is not None
    return estado


def prueba_tarjeta_sin_historia() -> None:
    caracteristicas = calcular_caracteristicas(None, 1_000, 50.0, "ropa")

    assert caracteristicas.cantidad_transacciones_24h == 0
    assert caracteristicas.monto_acumulado_tarjeta is None
    assert caracteristicas.ratio_monto_promedio_tarjeta is None
    assert caracteristicas.categoria_nueva_para_tarjeta is True


def prueba_ventanas_de_tiempo_son_estrictamente_anteriores() -> None:
    estado = _aplicar(_tx("a", 0), _tx("b", 300))

    # A 20 minutos de "a" y 15 de "b": "a" quedó fuera de la ventana de 10 min, "b" no.
    en_20_min = calcular_caracteristicas(estado, 1_200, 10.0, "comida")
    assert en_20_min.cantidad_transacciones_10min == 0
    assert en_20_min.cantidad_transacciones_1h == 2
    # Justo 10 min después de "b": el borde inferior de la ventana es inclusivo.
    en_borde = calcular_caracteristicas(estado, 900, 10.0, "comida")
    assert en_borde.cantidad_transacciones_10min == 1


def prueba_transacciones_del_mismo_segundo_no_se_ven_entre_si() -> None:
    estado = _aplicar(_tx("a", 100, monto=10.0), _tx("b", 500, monto=20.0))

    caracteristicas = calcular_caracteristicas(estado, 500, 40.0, "comida")

    assert caracteristicas.cantidad_transacciones_10min == 1
    assert caracteristicas.monto_acumulado_tarjeta == 10.0
    assert caracteristicas.ratio_monto_promedio_tarjeta == pytest.approx(4.0)


def prueba_categoria_nueva() -> None:
    estado = _aplicar(_tx("a", 0, categoria="comida"))

    assert calcular_caracteristicas(estado, 10, 1.0, "comida").categoria_nueva_para_tarjeta is False
    assert calcular_caracteristicas(estado, 10, 1.0, "ropa").categoria_nueva_para_tarjeta is True


def prueba_actualizar_es_idempotente_ante_reintentos() -> None:
    estado = _aplicar(_tx("a", 0), _tx("b", 60))

    assert actualizar_estado(estado, _tx("b", 60)) == estado
    assert actualizar_estado(estado, _tx("a", 0)) == estado


def prueba_el_estado_descarta_lo_que_salio_de_la_ventana_de_24_horas() -> None:
    estado = _aplicar(_tx("a", 0, monto=10.0), _tx("b", SEGUNDOS_24_HORAS + 1, monto=30.0))

    assert estado.marcas_recientes == (SEGUNDOS_24_HORAS + 1,)
    # Pero sigue contando en el acumulado y en el promedio histórico.
    assert estado.cantidad_total == 2
    assert estado.monto_total == 40.0


def prueba_rechaza_una_solicitud_anterior_al_estado() -> None:
    estado = _aplicar(_tx("a", 1_000))

    with pytest.raises(ValueError, match="anterior a la última"):
        calcular_caracteristicas(estado, 999, 10.0, "comida")


ESQUEMA = StructType(
    [
        StructField("id_transaccion", StringType()),
        StructField("numero_tarjeta", LongType()),
        StructField("marca_tiempo_unix", LongType()),
        StructField("categoria", StringType()),
        StructField("monto", DoubleType()),
    ]
)

COLUMNAS_CARACTERISTICAS = [
    "cantidad_transacciones_10min",
    "cantidad_transacciones_1h",
    "cantidad_transacciones_24h",
    "monto_acumulado_tarjeta",
    "ratio_monto_promedio_tarjeta",
    "categoria_nueva_para_tarjeta",
]


def _transacciones_aleatorias() -> pd.DataFrame:
    """Transacciones sintéticas que fuerzan los casos borde de las ventanas.

    Pocas tarjetas y muchas transacciones, con tres tipos de separación entre transacciones
    de una misma tarjeta: 0 segundos (empates), de minutos (llenan las ventanas cortas) y de
    más de un día (sacan cosas de la ventana de 24 h).
    """
    generador = np.random.default_rng(seed=7)
    filas: list[dict[str, object]] = []
    for tarjeta in (111, 222, 333, 444):
        marca = 1_600_000_000
        for _ in range(150):
            marca += int(
                generador.choice(
                    [0, 0, 1, 90, 400, 3_000, 20_000, 90_000],
                    p=[0.15, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.05],
                )
            )
            filas.append(
                {
                    "id_transaccion": f"tx_{len(filas):05d}",
                    "numero_tarjeta": tarjeta,
                    "marca_tiempo_unix": marca,
                    "categoria": str(
                        generador.choice(["comida", "ropa", "hogar", "ocio", "viajes"])
                    ),
                    "monto": float(np.round(generador.uniform(1, 500), 2)),
                }
            )
    return pd.DataFrame(filas)


def prueba_skew_online_coincide_con_el_batch(sesion_spark: SparkSession) -> None:
    """Training-serving skew (regla 3): el batch y la ruta online dan lo mismo, fila a fila."""
    transacciones = _transacciones_aleatorias()
    batch = (
        agregar_caracteristicas_historicas(sesion_spark.createDataFrame(transacciones, ESQUEMA))
        .toPandas()
        .set_index("id_transaccion")
    )

    online: dict[str, dict[str, object]] = {}
    orden = transacciones.sort_values(["numero_tarjeta", "marca_tiempo_unix", "id_transaccion"])
    for _, tarjeta in orden.groupby("numero_tarjeta"):
        estado: EstadoTarjeta | None = None
        for fila in tarjeta.to_dict("records"):
            transaccion = TransaccionTarjeta(
                fila["id_transaccion"], fila["marca_tiempo_unix"], fila["monto"], fila["categoria"]
            )
            caracteristicas = calcular_caracteristicas(
                estado, transaccion.marca_tiempo, transaccion.monto, transaccion.categoria
            )
            online[transaccion.id_transaccion] = vars(caracteristicas)
            estado = actualizar_estado(estado, transaccion)

    assert len(online) == len(batch) == 600
    for id_transaccion, esperadas in batch[COLUMNAS_CARACTERISTICAS].iterrows():
        obtenidas = online[str(id_transaccion)]
        for columna in COLUMNAS_CARACTERISTICAS:
            esperado, obtenido = esperadas[columna], obtenidas[columna]
            if pd.isna(esperado):
                assert obtenido is None, (id_transaccion, columna)
            else:
                assert obtenido == pytest.approx(esperado, rel=1e-9), (id_transaccion, columna)
