"""Arranque en frío del almacén online: carga en Redis el estado por tarjeta al corte de train.

Patrón real de producción: el lote (`fraude.lotes.calcular_estado_inicial`) aporta la historia
y el streaming solo aplica lo nuevo. `materialize` copia a Redis la fila más reciente de cada
tarjeta desde el Parquet del almacén offline.

Uso: `make arranque-en-frio` (necesita Redis: `make tiempo-real-arriba`).
"""

import logging
from datetime import UTC, datetime
from pathlib import Path

from feast import FeatureStore, Project

from fraude.caracteristicas.definiciones import (
    NOMBRE_VISTA_ESTADO_TARJETA,
    PROYECTO,
    TARJETA,
    crear_fuente_estado_inicial,
    crear_vista_estado_tarjeta,
)
from fraude.caracteristicas.tienda import crear_tienda_online

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RUTA_ESTADO_INICIAL = Path("datos_features/estado_tarjeta_inicial.parquet")
RUTA_REGISTRO = Path("datos_features/registro_feast.db")


def materializar_estado_inicial(
    tienda: FeatureStore, ruta_parquet: Path, proyecto: Project = PROYECTO
) -> None:
    """Da de alta las definiciones y materializa el Parquet del estado inicial en Redis."""
    vista = crear_vista_estado_tarjeta(crear_fuente_estado_inicial(ruta_parquet))
    tienda.apply([proyecto, TARJETA, vista])
    # El rango cubre cualquier fecha posible del dataset: `materialize` toma, por tarjeta,
    # la fila más reciente dentro de él.
    tienda.materialize(
        start_date=datetime(2000, 1, 1, tzinfo=UTC),
        end_date=datetime.now(UTC),
        feature_views=[NOMBRE_VISTA_ESTADO_TARJETA],
    )


def main() -> None:
    tienda = crear_tienda_online(RUTA_REGISTRO)
    materializar_estado_inicial(tienda, RUTA_ESTADO_INICIAL)
    logger.info("Estado inicial materializado en Redis desde %s", RUTA_ESTADO_INICIAL)


if __name__ == "__main__":
    main()
