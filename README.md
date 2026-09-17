# Detección de fraude con tarjetas en tiempo real

Puntúa transacciones con tarjeta en milisegundos a medida que llegan, con características en
streaming, feature store, reentrenamiento automático con etiquetas demoradas y monitoreo de drift.

> En desarrollo.

## Arquitectura

_Diagrama: Kafka → Spark Structured Streaming → Feast (Redis) → FastAPI en Kubernetes, con
características históricas en PySpark, MLflow, reentrenamiento con Airflow y
Prometheus/Grafana/Evidently._

## Decisiones de diseño

- **Dataset:** "Credit Card Transactions Fraud Detection Dataset" (generado con Sparkov,
  [kartik2112/fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection) en
  Kaggle). 1.852.394 transacciones, ~0,5% de fraude. Las columnas originales del CSV se renombran
  al español al cargarlas (ver `src/fraude/entrenamiento/carga.py`).
- **Partición temporal en tres partes:** Kaggle ya separa `fraudTrain.csv` (2019-01 a 2020-06) de
  `fraudTest.csv` (2020-06 a 2020-12) de forma cronológica. `fraudTrain.csv` se corta además en
  entrenamiento (hasta 2019-10) y validación (2019-11 en adelante), para poder elegir el umbral de
  costo y comparar campeón/retador sin tocar nunca el conjunto de prueba final
  (`src/fraude/entrenamiento/division.py`). Nunca se mezcla ni se muestrea al azar.
- **EDA en notebook versionado con jupytext:** `notebooks/01_analisis_exploratorio.py` (formato
  `.py` porcentaje) en vez de `.ipynb`, para que los diffs de git sean legibles. Queda fuera del
  lint/mypy estricto de `src/` y `pruebas/` porque tiene idioms propios de Jupyter (expresiones
  sueltas para mostrar resultados).

## Resultados

| Métrica | Reglas (línea base) | Gradient boosting | PyTorch |
|---------|---------------------|-------------------|---------|
| PR-AUC | _Pendiente_ | _Pendiente_ | _Pendiente_ |
| Costo de negocio | _Pendiente_ | _Pendiente_ | _Pendiente_ |
| Latencia p99 | — | _Pendiente_ | _Pendiente_ |

## Cómo ejecutarlo en local

```bash
cp .env.ejemplo .env
make instalar
make verificar
```

### Datos

El dataset no se versiona. Se descarga de Kaggle (requiere una cuenta y un token de la API en
`~/.kaggle/access_token`):

```bash
uv tool install kaggle
kaggle datasets download -d kartik2112/fraud-detection -p datos/ --unzip
```

Para abrir el notebook de EDA: `uv run jupyter lab notebooks/` (jupytext lo abre como notebook).

## Presupuesto de recursos

Desarrollado en una máquina de 8 GB: cada subsistema corre por separado en local y la integración
completa se hace en GitHub Codespaces. Aquí se documentará la memoria medida de cada componente.
