"""Lógica de un micro-lote del streaming: actualizar en Redis el estado de las tarjetas.

Separada de Spark a propósito: Spark solo entrega las filas de cada lote (`foreachBatch`), y
lo que decide qué le pasa al estado es esto, que se prueba sin Spark ni Kafka.

El estado vive **solo en Redis**. Por cada lote se lee el estado de las tarjetas involucradas,
se aplican sus transacciones en orden y se escribe el resultado (leer-modificar-escribir). Como
Structured Streaming entrega cada lote *al menos una vez*, un lote puede repetirse tras una
falla; eso es seguro porque `actualizar_estado` ignora las transacciones ya aplicadas.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from feast import FeatureStore

from fraude.caracteristicas.almacen_estado import escribir_estados, leer_estados
from fraude.caracteristicas.estado_tarjeta import (
    EstadoTarjeta,
    TransaccionTarjeta,
    actualizar_estado,
)


@dataclass(frozen=True)
class TransaccionEntrante:
    """Una transacción leída de Kafka, con la tarjeta a la que pertenece."""

    numero_tarjeta: int
    transaccion: TransaccionTarjeta


@dataclass(frozen=True)
class ResumenLote:
    transacciones: int
    tarjetas: int
    tarjetas_actualizadas: int


def aplicar_transacciones(
    previos: Mapping[int, EstadoTarjeta | None], entrantes: Sequence[TransaccionEntrante]
) -> dict[int, EstadoTarjeta]:
    """Aplica las transacciones a los estados previos; devuelve solo los que cambiaron.

    Dentro de cada tarjeta se ordena por (marca, id): es el orden en que `actualizar_estado`
    espera recibirlas. Kafka ya conserva ese orden por partición, pero no se confía en él
    para la corrección, solo para que en la práctica no haga falta reordenar. Una
    transacción ya aplicada no cambia nada, así que una tarjeta cuyo lote entero ya estaba
    aplicado no aparece en el resultado (y no se vuelve a escribir).
    """
    por_tarjeta: dict[int, list[TransaccionTarjeta]] = defaultdict(list)
    for entrante in entrantes:
        por_tarjeta[entrante.numero_tarjeta].append(entrante.transaccion)

    cambiados: dict[int, EstadoTarjeta] = {}
    for tarjeta, transacciones in por_tarjeta.items():
        previo = previos.get(tarjeta)
        estado = previo
        for transaccion in sorted(transacciones, key=lambda t: (t.marca_tiempo, t.id_transaccion)):
            estado = actualizar_estado(estado, transaccion)
        if estado is not None and estado != previo:
            cambiados[tarjeta] = estado
    return cambiados


def procesar_lote(tienda: FeatureStore, entrantes: Sequence[TransaccionEntrante]) -> ResumenLote:
    """Lee el estado de las tarjetas del lote, lo actualiza y lo escribe en el almacén online."""
    tarjetas = sorted({entrante.numero_tarjeta for entrante in entrantes})
    previos = leer_estados(tienda, tarjetas)
    cambiados = aplicar_transacciones(previos, entrantes)
    escribir_estados(tienda, cambiados)
    return ResumenLote(
        transacciones=len(entrantes),
        tarjetas=len(tarjetas),
        tarjetas_actualizadas=len(cambiados),
    )
