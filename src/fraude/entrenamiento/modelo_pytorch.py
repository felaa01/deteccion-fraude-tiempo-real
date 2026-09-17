"""Red neuronal en PyTorch con embeddings, como modelo retador frente a LightGBM.

El desbalance de clases se maneja con `pos_weight` en `BCEWithLogitsLoss` (el equivalente
en PyTorch al `class_weight="balanced"` de LightGBM), nunca remuestreando. La selección de
modelo entre épocas usa el costo de negocio en validación (nunca el conjunto de prueba),
igual que el umbral de decisión.
"""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fraude.entrenamiento.costo import elegir_umbral
from fraude.entrenamiento.modelo import COLUMNAS_CATEGORICAS, COLUMNAS_NUMERICAS

DIMENSIONES_EMBEDDING = {"categoria": 6, "genero": 2}
INDICE_DESCONOCIDO = 0


@dataclass(frozen=True)
class Preprocesador:
    """Estadísticos y vocabularios ajustados solo sobre el conjunto de entrenamiento."""

    medias: npt.NDArray[np.float64]
    desvios: npt.NDArray[np.float64]
    vocabularios: dict[str, dict[str, int]] = field(default_factory=dict)

    def tamanios_vocabulario(self) -> dict[str, int]:
        """Cardinalidad de cada categórica más una entrada extra para valores desconocidos."""
        return {columna: len(vocabulario) + 1 for columna, vocabulario in self.vocabularios.items()}


def preprocesador_a_dict(preprocesador: Preprocesador) -> dict[str, object]:
    """Representación JSON-serializable del preprocesador (para loguearlo como artefacto)."""
    return {
        "medias": preprocesador.medias.tolist(),
        "desvios": preprocesador.desvios.tolist(),
        "vocabularios": preprocesador.vocabularios,
    }


def ajustar_preprocesador(transacciones_entrenamiento: pd.DataFrame) -> Preprocesador:
    """Calcula media/desvío de las numéricas y el vocabulario de las categóricas en train."""
    numericas = transacciones_entrenamiento[COLUMNAS_NUMERICAS].to_numpy(dtype=np.float64)
    medias = numericas.mean(axis=0)
    desvios = numericas.std(axis=0)
    desvios[desvios == 0] = 1.0

    vocabularios = {
        columna: {
            valor: indice + 1
            for indice, valor in enumerate(sorted(transacciones_entrenamiento[columna].unique()))
        }
        for columna in COLUMNAS_CATEGORICAS
    }
    return Preprocesador(medias=medias, desvios=desvios, vocabularios=vocabularios)


def _transformar_numericas(
    preprocesador: Preprocesador, transacciones: pd.DataFrame
) -> torch.Tensor:
    numericas = transacciones[COLUMNAS_NUMERICAS].to_numpy(dtype=np.float64)
    estandarizadas = (numericas - preprocesador.medias) / preprocesador.desvios
    return torch.tensor(estandarizadas, dtype=torch.float32)


def _transformar_categoricas(
    preprocesador: Preprocesador, transacciones: pd.DataFrame
) -> dict[str, torch.Tensor]:
    return {
        columna: torch.tensor(
            transacciones[columna]
            .map(preprocesador.vocabularios[columna])
            .fillna(INDICE_DESCONOCIDO)
            .astype(np.int64)
            .to_numpy(),
            dtype=torch.long,
        )
        for columna in COLUMNAS_CATEGORICAS
    }


class ModeloFraudePyTorch(nn.Module):
    """MLP con embeddings para las categóricas, concatenados con las numéricas estandarizadas."""

    def __init__(self, cantidad_numericas: int, tamanios_vocabulario: dict[str, int]) -> None:
        super().__init__()
        self.embeddings = nn.ModuleDict(
            {
                columna: nn.Embedding(tamanio, DIMENSIONES_EMBEDDING[columna])
                for columna, tamanio in tamanios_vocabulario.items()
            }
        )
        entrada = cantidad_numericas + sum(DIMENSIONES_EMBEDDING[c] for c in COLUMNAS_CATEGORICAS)
        self.mlp = nn.Sequential(
            nn.Linear(entrada, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self, numericas: torch.Tensor, categoricas: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Devuelve el logit de fraude (sin sigmoide) para cada fila."""
        embebidas = [self.embeddings[columna](categoricas[columna]) for columna in categoricas]
        entrada = torch.cat([numericas, *embebidas], dim=1)
        salida: torch.Tensor = self.mlp(entrada).squeeze(1)
        return salida


def entrenar_red(
    entrenamiento: pd.DataFrame,
    validacion: pd.DataFrame,
    monto_validacion: npt.NDArray[np.float64],
    costo_revision: float,
    semilla: int = 0,
    epocas: int = 15,
    tamanio_lote: int = 2048,
) -> tuple[ModeloFraudePyTorch, Preprocesador]:
    """Entrena la red y se queda con los pesos de la época de menor costo en validación."""
    torch.manual_seed(semilla)
    preprocesador = ajustar_preprocesador(entrenamiento)

    x_numerico = _transformar_numericas(preprocesador, entrenamiento)
    x_categorico = _transformar_categoricas(preprocesador, entrenamiento)
    y = torch.tensor(entrenamiento["es_fraude"].to_numpy(dtype=np.float32))

    x_numerico_validacion = _transformar_numericas(preprocesador, validacion)
    x_categorico_validacion = _transformar_categoricas(preprocesador, validacion)
    es_fraude_validacion = validacion["es_fraude"].astype(bool).to_numpy()

    modelo = ModeloFraudePyTorch(len(COLUMNAS_NUMERICAS), preprocesador.tamanios_vocabulario())

    cantidad_positivos = y.sum().item()
    pos_weight = torch.tensor((len(y) - cantidad_positivos) / max(cantidad_positivos, 1.0))
    perdida = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)

    conjunto = TensorDataset(x_numerico, x_categorico["categoria"], x_categorico["genero"], y)
    cargador = DataLoader(
        conjunto,
        batch_size=tamanio_lote,
        shuffle=True,
        generator=torch.Generator().manual_seed(semilla),
    )

    mejor_costo = float("inf")
    mejores_pesos = {nombre: tensor.clone() for nombre, tensor in modelo.state_dict().items()}

    for _epoca in range(epocas):
        modelo.train()
        for lote_numerico, lote_categoria, lote_genero, lote_y in cargador:
            optimizador.zero_grad()
            logits = modelo(lote_numerico, {"categoria": lote_categoria, "genero": lote_genero})
            perdida(logits, lote_y).backward()
            optimizador.step()

        modelo.eval()
        with torch.no_grad():
            logits_validacion = modelo(x_numerico_validacion, x_categorico_validacion)
            probabilidad_validacion = torch.sigmoid(logits_validacion).numpy()
        costo_epoca = elegir_umbral(
            es_fraude_validacion,
            probabilidad_validacion,
            monto_validacion,
            costo_revision=costo_revision,
        ).costo
        if costo_epoca < mejor_costo:
            mejor_costo = costo_epoca
            mejores_pesos = {
                nombre: tensor.clone() for nombre, tensor in modelo.state_dict().items()
            }

    modelo.load_state_dict(mejores_pesos)
    return modelo, preprocesador


def predecir_probabilidad(
    modelo: ModeloFraudePyTorch, preprocesador: Preprocesador, transacciones: pd.DataFrame
) -> npt.NDArray[np.float64]:
    """Probabilidad de fraude que asigna la red a cada transacción."""
    modelo.eval()
    with torch.no_grad():
        numericas = _transformar_numericas(preprocesador, transacciones)
        categoricas = _transformar_categoricas(preprocesador, transacciones)
        logits = modelo(numericas, categoricas)
        probabilidades: npt.NDArray[np.float64] = torch.sigmoid(logits).numpy().astype(np.float64)
    return probabilidades
