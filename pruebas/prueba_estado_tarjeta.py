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
from fraude.caracteristicas.geografia import distancia_haversine_km
from fraude.lotes.caracteristicas_historicas import agregar_caracteristicas_historicas


def _tx(
    id_transaccion: str,
    marca: int,
    monto: float = 10.0,
    categoria: str = "comida",
    latitud: float = 0.0,
    longitud: float = 0.0,
) -> TransaccionTarjeta:
    return TransaccionTarjeta(id_transaccion, marca, monto, categoria, latitud, longitud)


def _aplicar(*transacciones: TransaccionTarjeta) -> EstadoTarjeta:
    estado: EstadoTarjeta | None = None
    for transaccion in transacciones:
        estado = actualizar_estado(estado, transaccion)
    assert estado is not None
    return estado


def prueba_tarjeta_sin_historia() -> None:
    caracteristicas = calcular_caracteristicas(None, 1_000, 50.0, "ropa", 0.0, 0.0)

    assert caracteristicas.cantidad_transacciones_24h == 0
    assert caracteristicas.monto_acumulado_tarjeta is None
    assert caracteristicas.ratio_monto_promedio_tarjeta is None
    assert caracteristicas.categoria_nueva_para_tarjeta is True
    assert caracteristicas.distancia_transaccion_anterior_km is None
    assert caracteristicas.velocidad_implicita_kmh is None


def prueba_ventanas_de_tiempo_son_estrictamente_anteriores() -> None:
    estado = _aplicar(_tx("a", 0), _tx("b", 300))

    # A 20 minutos de "a" y 15 de "b": "a" quedó fuera de la ventana de 10 min, "b" no.
    en_20_min = calcular_caracteristicas(estado, 1_200, 10.0, "comida", 0.0, 0.0)
    assert en_20_min.cantidad_transacciones_10min == 0
    assert en_20_min.cantidad_transacciones_1h == 2
    # Justo 10 min después de "b": el borde inferior de la ventana es inclusivo.
    en_borde = calcular_caracteristicas(estado, 900, 10.0, "comida", 0.0, 0.0)
    assert en_borde.cantidad_transacciones_10min == 1


def prueba_transacciones_del_mismo_segundo_no_se_ven_entre_si() -> None:
    estado = _aplicar(_tx("a", 100, monto=10.0), _tx("b", 500, monto=20.0))

    caracteristicas = calcular_caracteristicas(estado, 500, 40.0, "comida", 0.0, 0.0)

    assert caracteristicas.cantidad_transacciones_10min == 1
    assert caracteristicas.monto_acumulado_tarjeta == 10.0
    assert caracteristicas.ratio_monto_promedio_tarjeta == pytest.approx(4.0)


def prueba_categoria_nueva() -> None:
    estado = _aplicar(_tx("a", 0, categoria="comida"))

    para_comida = calcular_caracteristicas(estado, 10, 1.0, "comida", 0.0, 0.0)
    para_ropa = calcular_caracteristicas(estado, 10, 1.0, "ropa", 0.0, 0.0)

    assert para_comida.categoria_nueva_para_tarjeta is False
    assert para_ropa.categoria_nueva_para_tarjeta is True


def prueba_distancia_y_velocidad_respecto_de_la_transaccion_anterior() -> None:
    estado = _aplicar(_tx("a", 0, latitud=0.0, longitud=0.0))

    # Un grado de latitud (~111,19 km) una hora después: ~111,19 km/h.
    caracteristicas = calcular_caracteristicas(estado, 3_600, 10.0, "comida", 1.0, 0.0)

    esperada = distancia_haversine_km(0.0, 0.0, 1.0, 0.0)
    assert esperada == pytest.approx(111.19, abs=0.01)
    assert caracteristicas.distancia_transaccion_anterior_km == pytest.approx(esperada)
    assert caracteristicas.velocidad_implicita_kmh == pytest.approx(esperada)


def prueba_la_transaccion_anterior_es_la_ultima_aplicada() -> None:
    estado = _aplicar(
        _tx("a", 0, latitud=0.0, longitud=0.0), _tx("b", 60, latitud=0.0, longitud=1.0)
    )

    caracteristicas = calcular_caracteristicas(estado, 120, 10.0, "comida", 0.0, 1.0)

    # Se mide contra "b" (0, 1), no contra "a": misma ubicación que "b".
    assert caracteristicas.distancia_transaccion_anterior_km == pytest.approx(0.0, abs=1e-9)


def prueba_con_el_mismo_segundo_hay_distancia_pero_no_velocidad() -> None:
    estado = _aplicar(_tx("a", 100, latitud=0.0, longitud=0.0))

    caracteristicas = calcular_caracteristicas(estado, 100, 10.0, "comida", 1.0, 0.0)

    assert caracteristicas.distancia_transaccion_anterior_km == pytest.approx(111.19, abs=0.01)
    assert caracteristicas.velocidad_implicita_kmh is None


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


def prueba_el_estado_guarda_la_ubicacion_de_la_ultima_transaccion() -> None:
    estado = _aplicar(
        _tx("a", 0, latitud=10.0, longitud=20.0), _tx("b", 60, latitud=30.0, longitud=40.0)
    )

    assert (estado.ultima_latitud_comercio, estado.ultima_longitud_comercio) == (30.0, 40.0)


def prueba_rechaza_una_solicitud_anterior_al_estado() -> None:
    estado = _aplicar(_tx("a", 1_000))

    with pytest.raises(ValueError, match="anterior a la última"):
        calcular_caracteristicas(estado, 999, 10.0, "comida", 0.0, 0.0)


ESQUEMA = StructType(
    [
        StructField("id_transaccion", StringType()),
        StructField("numero_tarjeta", LongType()),
        StructField("marca_tiempo", LongType()),
        StructField("categoria", StringType()),
        StructField("monto", DoubleType()),
        StructField("latitud_comercio", DoubleType()),
        StructField("longitud_comercio", DoubleType()),
    ]
)

COLUMNAS_CARACTERISTICAS = [
    "cantidad_transacciones_10min",
    "cantidad_transacciones_1h",
    "cantidad_transacciones_24h",
    "monto_acumulado_tarjeta",
    "ratio_monto_promedio_tarjeta",
    "categoria_nueva_para_tarjeta",
    "distancia_transaccion_anterior_km",
    "velocidad_implicita_kmh",
]


def prueba_skew_online_coincide_con_el_batch(
    sesion_spark: SparkSession, transacciones_aleatorias: pd.DataFrame
) -> None:
    """Training-serving skew (regla 3): el batch y la ruta online dan lo mismo, fila a fila."""
    transacciones = transacciones_aleatorias
    batch = (
        agregar_caracteristicas_historicas(sesion_spark.createDataFrame(transacciones, ESQUEMA))
        .toPandas()
        .set_index("id_transaccion")
    )

    online: dict[str, dict[str, object]] = {}
    orden = transacciones.sort_values(["numero_tarjeta", "marca_tiempo", "id_transaccion"])
    for _, tarjeta in orden.groupby("numero_tarjeta"):
        estado: EstadoTarjeta | None = None
        for fila in tarjeta.to_dict("records"):
            transaccion = TransaccionTarjeta(
                fila["id_transaccion"],
                fila["marca_tiempo"],
                fila["monto"],
                fila["categoria"],
                fila["latitud_comercio"],
                fila["longitud_comercio"],
            )
            caracteristicas = calcular_caracteristicas(
                estado,
                transaccion.marca_tiempo,
                transaccion.monto,
                transaccion.categoria,
                transaccion.latitud_comercio,
                transaccion.longitud_comercio,
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
