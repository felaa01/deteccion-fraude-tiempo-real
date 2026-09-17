# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (fraude)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Análisis exploratorio — transacciones de tarjetas (dataset Sparkov de Kaggle)
#
# Primer vistazo a `fraudTrain.csv` / `fraudTest.csv` antes de armar la línea base.
# Preguntas que nos importan para el diseño de características y de la partición:
#
# - ¿Cómo es el desbalance de clases en cada conjunto?
# - ¿Cómo varía el monto entre transacciones legítimas y fraudulentas?
# - ¿El fraude se concentra en ciertas categorías de comercio u horarios?
# - ¿Cuántas transacciones previas tiene una tarjeta típica antes del corte de
#   validación? (afecta a las características de velocidad, que necesitan historia)

# %%
from pathlib import Path

import matplotlib.pyplot as plt

import fraude
from fraude.entrenamiento.carga import cargar_transacciones
from fraude.entrenamiento.division import dividir_temporalmente

# Se resuelve a partir del paquete instalado (editable) en vez de una ruta relativa,
# porque el directorio de trabajo del kernel varía según cómo se lance Jupyter.
RUTA_REPO = Path(fraude.__file__).resolve().parents[2]
RUTA_DATOS = RUTA_REPO / "datos"

# %%
crudo_entrenamiento = cargar_transacciones(RUTA_DATOS / "fraudTrain.csv")
crudo_prueba = cargar_transacciones(RUTA_DATOS / "fraudTest.csv")
conjuntos = dividir_temporalmente(crudo_entrenamiento, crudo_prueba)

for nombre, datos in [
    ("entrenamiento", conjuntos.entrenamiento),
    ("validación", conjuntos.validacion),
    ("prueba", conjuntos.prueba),
]:
    tasa = datos["es_fraude"].mean() * 100
    print(f"{nombre}: {len(datos):,} filas, {tasa:.3f}% fraude")

# %% [markdown]
# ## Distribución del monto: fraude vs. legítimas

# %%
datos_entrenamiento = conjuntos.entrenamiento
fig, ejes = plt.subplots(1, 2, figsize=(10, 4))
for eje, es_fraude, titulo in [
    (ejes[0], 0, "Legítimas"),
    (ejes[1], 1, "Fraude"),
]:
    datos_entrenamiento.loc[datos_entrenamiento["es_fraude"] == es_fraude, "monto"].hist(
        bins=50, ax=eje
    )
    eje.set_title(titulo)
    eje.set_xlabel("monto")
plt.tight_layout()

# %% [markdown]
# ## Tasa de fraude por categoría de comercio

# %%
tasa_por_categoria = (
    datos_entrenamiento.groupby("categoria")["es_fraude"].mean().sort_values(ascending=False) * 100
)
tasa_por_categoria

# %% [markdown]
# ## Tasa de fraude por hora del día

# %%
por_hora = datos_entrenamiento.assign(
    hora=datos_entrenamiento["fecha_hora_transaccion"].dt.hour
).groupby("hora")["es_fraude"].mean() * 100
por_hora.plot(kind="bar", figsize=(10, 4))
plt.ylabel("% fraude")

# %% [markdown]
# ## Transacciones previas por tarjeta antes del corte de validación
#
# Si una tarjeta tiene pocas transacciones antes del corte, las características de
# velocidad (conteos en ventanas de tiempo) van a ser poco confiables para ella.

# %%
conteo_por_tarjeta = datos_entrenamiento.groupby("numero_tarjeta").size()
conteo_por_tarjeta.describe()
