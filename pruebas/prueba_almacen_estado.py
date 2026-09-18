from pathlib import Path

import pytest
from feast import FeatureStore, Project

from fraude.caracteristicas.almacen_estado import (
    escribir_estados,
    estados_a_dataframe,
    leer_estados,
)
from fraude.caracteristicas.definiciones import (
    COLUMNA_TIEMPO_ESTADO,
    TARJETA,
    crear_fuente_estado_inicial,
    crear_vista_estado_tarjeta,
)
from fraude.caracteristicas.estado_tarjeta import (
    EstadoTarjeta,
    TransaccionTarjeta,
    actualizar_estado,
)
from fraude.caracteristicas.tienda import crear_tienda_online

# Las pruebas de integración usan su propio proyecto de Feast y la base 1 de Redis (la base 0
# es la del estado real), y la vacían antes y después de cada prueba.
CONEXION_REDIS_DE_PRUEBA = "localhost:6379,db=1"
PROYECTO_DE_PRUEBA = "prueba_estado_tarjeta"
TARJETA_A = 9_000_000_000_000_001
TARJETA_B = 9_000_000_000_000_002
TARJETA_INEXISTENTE = 9_000_000_000_000_003


def _estado() -> EstadoTarjeta:
    estado = actualizar_estado(None, TransaccionTarjeta("tx_1", 1_600_000_000, 10.5, "comida"))
    estado = actualizar_estado(estado, TransaccionTarjeta("tx_2", 1_600_000_060, 20.25, "ropa"))
    return actualizar_estado(estado, TransaccionTarjeta("tx_3", 1_600_000_060, 5.0, "comida"))


def _tienda_de_prueba(directorio: Path) -> FeatureStore:
    tienda = crear_tienda_online(
        directorio / "registro.db", CONEXION_REDIS_DE_PRUEBA, proyecto=PROYECTO_DE_PRUEBA
    )
    vista = crear_vista_estado_tarjeta(crear_fuente_estado_inicial(directorio / "inicial.parquet"))
    tienda.apply([Project(name=PROYECTO_DE_PRUEBA), TARJETA, vista])
    return tienda


def prueba_dataframe_tiene_una_fila_por_tarjeta_con_la_columna_de_tiempo() -> None:
    estado = _estado()

    df = estados_a_dataframe({TARJETA_A: estado})

    assert df["numero_tarjeta"].tolist() == [TARJETA_A]
    assert df[COLUMNA_TIEMPO_ESTADO].iloc[0].timestamp() == estado.ultima_marca
    assert df["categorias_vistas"].iloc[0] == ["comida", "ropa"]


@pytest.mark.integracion
def prueba_ida_y_vuelta_por_redis(tmp_path: Path, redis_de_prueba: None) -> None:
    """Escribe el estado con `push` y lo lee igual, incluido el de una tarjeta desconocida."""
    tienda = _tienda_de_prueba(tmp_path)
    estado_a = _estado()
    estado_b = actualizar_estado(None, TransaccionTarjeta("tx_9", 1_600_000_100, 99.99, "ocio"))

    escribir_estados(tienda, {TARJETA_A: estado_a, TARJETA_B: estado_b})
    leidos = leer_estados(tienda, [TARJETA_A, TARJETA_B, TARJETA_INEXISTENTE])

    assert leidos[TARJETA_A] == estado_a
    assert leidos[TARJETA_B] == estado_b
    assert leidos[TARJETA_INEXISTENTE] is None


@pytest.mark.integracion
def prueba_dos_escrituras_con_el_mismo_timestamp_no_se_pierden(
    tmp_path: Path, redis_de_prueba: None
) -> None:
    """Regresión de `skip_dedup`: Feast descartaría la segunda por tener el mismo instante."""
    tienda = _tienda_de_prueba(tmp_path)
    primero = actualizar_estado(None, TransaccionTarjeta("tx_a", 1_600_000_000, 10.0, "comida"))
    segundo = actualizar_estado(primero, TransaccionTarjeta("tx_b", 1_600_000_000, 20.0, "comida"))

    escribir_estados(tienda, {TARJETA_A: primero})
    escribir_estados(tienda, {TARJETA_A: segundo})

    leido = leer_estados(tienda, [TARJETA_A])[TARJETA_A]
    assert leido == segundo
    assert leido is not None
    assert leido.cantidad_total == 2
