"""Construcción de la `FeatureStore` de Feast usada por el proyecto.

Se arma con un `RepoConfig` en código en vez de un `feature_store.yaml`: no hay ninguna
razón para depender de un archivo con rutas relativas cuando el proyecto ya sabe, en
Python, dónde viven el registro y los datos.
"""

from pathlib import Path

from feast import FeatureStore
from feast.repo_config import RepoConfig


def crear_tienda(ruta_registro: Path, ruta_almacen_online: Path) -> FeatureStore:
    """Crea la `FeatureStore` del proyecto.

    El almacén online (sqlite) todavía no se usa para nada -- Feast igual lo exige para
    poder aplicar las definiciones. El almacén offline es Parquet (`type: file`).
    """
    ruta_registro.parent.mkdir(parents=True, exist_ok=True)
    ruta_almacen_online.parent.mkdir(parents=True, exist_ok=True)
    configuracion = RepoConfig(
        project="deteccion_fraude",
        provider="local",
        registry=str(ruta_registro),
        offline_store={"type": "file"},
        online_store={"type": "sqlite", "path": str(ruta_almacen_online)},
    )
    return FeatureStore(config=configuracion)


def crear_tienda_online(
    ruta_registro: Path,
    conexion_redis: str = "localhost:6379",
    proyecto: str = "deteccion_fraude",
) -> FeatureStore:
    """Crea la `FeatureStore` con Redis como almacén online.

    `skip_dedup=True` es a propósito. Por defecto, el almacén Redis de Feast descarta un
    write cuyo timestamp sea menor **o igual** al ya guardado; como el timestamp del estado
    es la fecha de la última transacción, dos transacciones de una tarjeta en el mismo
    segundo harían que la segunda se perdiera en silencio. La protección que la
    deduplicación daría (que un valor viejo no pise a uno nuevo) ya la da
    `actualizar_estado`, que es idempotente y compara (marca, id) antes de cambiar nada, y
    hay un único escritor.
    """
    ruta_registro.parent.mkdir(parents=True, exist_ok=True)
    configuracion = RepoConfig(
        project=proyecto,
        provider="local",
        registry=str(ruta_registro),
        offline_store={"type": "file"},
        online_store={
            "type": "redis",
            "connection_string": conexion_redis,
            "skip_dedup": True,
        },
    )
    return FeatureStore(config=configuracion)
