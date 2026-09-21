"""Ritmo de publicación: cuándo hay que enviar cada mensaje.

La reproducción es **acelerada**: `aceleracion` segundos de tiempo simulado transcurren por
cada segundo real. Con `aceleracion=3600`, una hora de transacciones se publica en un
segundo y los ~6 meses de `fraudTest` en algo más de una hora.
"""

from datetime import datetime


def instante_de_envio(
    fecha_evento: datetime,
    fecha_inicial: datetime,
    inicio_real: float,
    aceleracion: float,
) -> float:
    """Instante real (en segundos del reloj monotónico) en que hay que publicar el evento.

    Se calcula siempre contra el *inicio* de la reproducción y no como "esperar la
    diferencia con el mensaje anterior": esperar diferencias acumula el error de cada
    `sleep` (siempre duerme un poco de más) y la reproducción se va atrasando. Con un
    instante absoluto por mensaje, si un envío se demora, los siguientes salen apenas
    antes y el ritmo se recupera solo.
    """
    if aceleracion <= 0:
        raise ValueError("La aceleración tiene que ser mayor que cero")
    segundos_simulados = (fecha_evento - fecha_inicial).total_seconds()
    return inicio_real + segundos_simulados / aceleracion
