"""Armado de los mensajes que el productor publica en Kafka.

Lo que viaja por el tópico es lo que un procesador de pagos real vería en el momento de
autorizar una transacción. Por eso el mensaje **no** incluye:

- `es_fraude`: la etiqueta llega días o semanas después, con el contracargo (regla 6). Si
  viajara en el mensaje, el streaming podría usarla sin querer.
- Datos personales que ninguna característica necesita (nombre, apellido, calle, ocupación,
  etc.).
- `marca_tiempo_unix` (`unix_time` del dataset): no es confiable como tiempo. Está unos 7 años
  atrás de `fecha_hora_transaccion` (2013 contra 2020) y en `fraudTrain` el desfase no es
  constante: 2557 días, 2556 entre el 2019-02-28 y el 2020-03-01. El tiempo del evento es siempre
  `fecha_hora_transaccion`, la misma que usan el almacén offline y el batch.
"""

import json
from collections.abc import Iterator, Mapping
from datetime import date, datetime
from typing import Any

import pandas as pd

CAMPOS_MENSAJE = (
    "id_transaccion",
    "numero_tarjeta",
    "fecha_hora_transaccion",
    "monto",
    "categoria",
    "comercio",
    "genero",
    "fecha_nacimiento",
    "latitud_cliente",
    "longitud_cliente",
    "latitud_comercio",
    "longitud_comercio",
    "poblacion_ciudad",
)


def preparar_transacciones(transacciones: pd.DataFrame) -> pd.DataFrame:
    """Deja solo los campos del mensaje, en el orden en que ocurrieron las transacciones.

    Ordena por fecha y desempata por `id_transaccion` para que el orden sea determinista:
    hay muchas transacciones con el mismo segundo y sin desempate el orden de publicación
    (y con él, el resultado de una comparación offline contra online) variaría entre
    corridas.
    """
    ordenadas = transacciones.sort_values(["fecha_hora_transaccion", "id_transaccion"])
    return ordenadas[list(CAMPOS_MENSAJE)].reset_index(drop=True)


def _a_iso(valor: object) -> str:
    if isinstance(valor, datetime | date):
        return valor.isoformat()
    raise TypeError(f"Tipo no serializable en un mensaje: {type(valor).__name__}")


def serializar(fila: Mapping[str, Any]) -> bytes:
    """Serializa una transacción a JSON (UTF-8). Las fechas van en formato ISO 8601."""
    return json.dumps(fila, default=_a_iso, separators=(",", ":")).encode("utf-8")


def clave_de_particion(fila: Mapping[str, Any]) -> bytes:
    """La clave del mensaje es la tarjeta.

    Kafka solo garantiza orden dentro de una partición y asigna la partición por el hash
    de la clave: con esta clave todas las transacciones de una tarjeta caen en la misma
    partición y se consumen en el orden en que se publicaron. Las características por
    ventana de tarjeta dependen de ese orden.
    """
    return str(fila["numero_tarjeta"]).encode("utf-8")


def iterar_filas(
    transacciones: pd.DataFrame, tamano_bloque: int = 10_000
) -> Iterator[dict[str, Any]]:
    """Recorre el DataFrame de a bloques, sin materializar todas las filas como dicts."""
    for inicio in range(0, len(transacciones), tamano_bloque):
        bloque = transacciones.iloc[inicio : inicio + tamano_bloque]
        yield from bloque.to_dict("records")  # type: ignore[misc]
