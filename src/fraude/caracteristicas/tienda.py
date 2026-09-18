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
