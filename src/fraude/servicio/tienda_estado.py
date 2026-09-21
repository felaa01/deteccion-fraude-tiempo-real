"""Arma la tienda de Feast que el servicio usa para leer el estado por tarjeta de Redis.

Se configura por variables de entorno (la imagen no sabe dónde está Redis):
- `REDIS_CONEXION`: `host:puerto` de Redis (por defecto `localhost:6379`).
- `REGISTRO_FEAST`: ruta del registro local de Feast (por defecto
  `datos_features/registro_feast.db`).

El servicio solo **lee**: el único escritor del estado es el job de Spark Streaming.
"""

import os
from pathlib import Path

from feast import FeatureStore

from fraude.caracteristicas.definiciones import (
    PROYECTO,
    TARJETA,
    crear_fuente_estado_inicial,
    crear_vista_estado_tarjeta,
)
from fraude.caracteristicas.tienda import crear_tienda_online

# La vista exige una fuente batch, pero el servicio solo lee de Redis: el Parquet no se abre.
_FUENTE_ESTADO_INICIAL = Path("datos_features/estado_tarjeta_inicial.parquet")


def cargar_tienda() -> FeatureStore:
    """Crea la tienda con Redis como almacén online y da de alta las definiciones.

    El `apply` solo escribe metadatos en el registro local del contenedor: no toca los datos
    de Redis. Así el servicio arranca sin depender de un registro generado en otra máquina.
    Las lecturas de Redis usan el nombre del proyecto y la vista, que tienen que coincidir
    con los que usó el job de streaming al escribir (ambos salen de `definiciones`).
    """
    ruta_registro = Path(os.environ.get("REGISTRO_FEAST", "datos_features/registro_feast.db"))
    conexion_redis = os.environ.get("REDIS_CONEXION", "localhost:6379")

    tienda = crear_tienda_online(ruta_registro, conexion_redis, PROYECTO.name)
    vista = crear_vista_estado_tarjeta(crear_fuente_estado_inicial(_FUENTE_ESTADO_INICIAL))
    tienda.apply([PROYECTO, TARJETA, vista])
    return tienda
