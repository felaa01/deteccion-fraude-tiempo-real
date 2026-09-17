from collections.abc import Iterator

import pytest
from pyspark.sql import SparkSession


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
