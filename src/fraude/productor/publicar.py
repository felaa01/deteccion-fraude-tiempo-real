"""Reproduce `fraudTest.csv` en un tópico de Kafka, en orden temporal y a velocidad acelerada.

Uso: `make productor` (o `uv run python -m fraude.productor.publicar --ayuda`).
"""

import argparse
import logging
import time
from pathlib import Path

from confluent_kafka import KafkaError, Message, Producer

from fraude.entrenamiento.carga import cargar_transacciones
from fraude.productor.mensajes import (
    clave_de_particion,
    iterar_filas,
    preparar_transacciones,
    serializar,
)
from fraude.productor.ritmo import instante_de_envio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RUTA_TRANSACCIONES = Path("datos/fraudTest.csv")
INTERVALO_LOG = 10_000


def _crear_productor(servidor: str) -> Producer:
    return Producer(
        {
            "bootstrap.servers": servidor,
            # Un mensaje de transacción no se puede perder ni duplicar: `acks=all` espera la
            # confirmación de todas las réplicas en sincronía y la idempotencia hace que un
            # reintento del cliente no escriba el mismo mensaje dos veces. Sobre un solo
            # nodo `acks=all` equivale a `acks=1`, pero es la configuración correcta para
            # producción y no cuesta nada dejarla.
            "acks": "all",
            "enable.idempotence": True,
            # Pequeño lote de 5 ms: agrupa mensajes sin agregar latencia perceptible.
            "linger.ms": 5,
        }
    )


class _Entregas:
    """Cuenta las confirmaciones y los errores que informa el broker."""

    def __init__(self) -> None:
        self.confirmadas = 0
        self.fallidas = 0

    def __call__(self, error: KafkaError | None, mensaje: Message) -> None:
        if error is not None:
            self.fallidas += 1
            logger.error("Falló la entrega de un mensaje: %s", error)
        else:
            self.confirmadas += 1


def publicar(
    ruta: Path,
    servidor: str,
    topico: str,
    aceleracion: float,
    limite: int | None,
) -> None:
    """Publica las transacciones de `ruta`. `aceleracion=0` publica sin esperas."""
    transacciones = preparar_transacciones(cargar_transacciones(ruta))
    if limite is not None:
        transacciones = transacciones.head(limite)
    total = len(transacciones)
    logger.info("Publicando %d transacciones en '%s' (aceleración x%g)", total, topico, aceleracion)

    productor = _crear_productor(servidor)
    entregas = _Entregas()
    fecha_inicial = transacciones["fecha_hora_transaccion"].iloc[0].to_pydatetime()
    inicio_real = time.monotonic()

    for numero, fila in enumerate(iterar_filas(transacciones), start=1):
        if aceleracion > 0:
            objetivo = instante_de_envio(
                fila["fecha_hora_transaccion"], fecha_inicial, inicio_real, aceleracion
            )
            espera = objetivo - time.monotonic()
            if espera > 0:
                time.sleep(espera)

        while True:
            try:
                productor.produce(
                    topico,
                    key=clave_de_particion(fila),
                    value=serializar(fila),
                    on_delivery=entregas,
                )
                break
            except BufferError:
                # La cola interna del cliente se llenó: dejar que el broker la drene.
                productor.poll(1.0)
        productor.poll(0)

        if numero % INTERVALO_LOG == 0:
            logger.info(
                "%d/%d publicadas (simulado: %s)", numero, total, fila["fecha_hora_transaccion"]
            )

    pendientes = productor.flush(30)
    logger.info(
        "Listo: %d confirmadas, %d fallidas, %d sin confirmar en %.1f s",
        entregas.confirmadas,
        entregas.fallidas,
        pendientes,
        time.monotonic() - inicio_real,
    )
    if entregas.fallidas or pendientes:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ruta", type=Path, default=RUTA_TRANSACCIONES)
    parser.add_argument("--servidor", default="localhost:9092")
    parser.add_argument("--topico", default="transacciones")
    parser.add_argument(
        "--aceleracion",
        type=float,
        default=3600.0,
        help="Segundos simulados por segundo real (0 = sin esperas, lo más rápido posible).",
    )
    parser.add_argument("--limite", type=int, default=None, help="Publicar solo las primeras N.")
    args = parser.parse_args()
    publicar(args.ruta, args.servidor, args.topico, args.aceleracion, args.limite)


if __name__ == "__main__":
    main()
