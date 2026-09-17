import fraude


def prueba_el_paquete_se_puede_importar() -> None:
    assert fraude.__doc__ == "Detección de fraude con tarjetas en tiempo real."
