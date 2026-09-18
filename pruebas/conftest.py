from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession
from redis import Redis


@pytest.fixture(scope="session")
def sesion_spark() -> Iterator[SparkSession]:
    """Sesión de Spark local, chica, para pruebas unitarias (sin servicios externos)."""
    sesion = (
        SparkSession.builder.appName("pruebas-fraude")
        .master("local[1]")
        .config("spark.driver.memory", "512m")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield sesion
    sesion.stop()


@pytest.fixture
def transacciones_aleatorias() -> pd.DataFrame:
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


@pytest.fixture
def redis_de_prueba() -> Iterator[None]:
    """Base 1 de Redis (la 0 es la del estado real), vaciada antes y después de la prueba."""
    cliente = Redis(host="localhost", port=6379, db=1)
    cliente.flushdb()
    yield
    cliente.flushdb()
