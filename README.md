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
- **Costo de negocio:** falso negativo = monto completo de la transacción (se pierde la plata);
  falso positivo o verdadero positivo = costo fijo de revisión, USD 10 (fricción con el cliente o
  investigación manual, se haya bloqueado a tiempo o no). Verdadero negativo = 0. Es el esquema
  estándar de la literatura de cost-sensitive fraud detection (Bahnsen et al.).
- **Características de la línea base ("sin historia"):** monto, categoría, hora del día, día de la
  semana, edad del titular y distancia cliente↔comercio (haversine sobre lat/long). Todas
  calculables por fila, sin necesitar transacciones previas de la tarjeta — eso se suma más
  adelante con las features de velocidad de Spark Streaming/Feast.
- **Gradient boosting: LightGBM**, no `sklearn.GradientBoostingClassifier`. Maneja categóricas de
  forma nativa, pesos por clase para el desbalance (`class_weight="balanced"`, nunca remuestrear) y
  es liviano en CPU sin GPU.
- **MLflow en Docker** (perfil `entrenamiento`), backend en SQLite y artefactos servidos vía proxy
  (`mlflow-artifacts:/` + `--artifacts-destination`), no con una ruta local directa: el cliente de
  MLflow corre en el host, no en el contenedor, así que necesita que el servidor intermedie la
  subida de artefactos en vez de escribir a un path que solo existe dentro del contenedor.

## Resultados

Evaluado sobre `fraudTest.csv` (conjunto de prueba final, nunca usado para elegir el umbral),
con la matriz de costo de [Decisiones de diseño](#decisiones-de-diseño):

| Métrica | Reglas (línea base) | Gradient boosting (LightGBM) | PyTorch |
|---------|---------------------|-------------------------------|---------|
| Umbral elegido en validación | monto ≥ USD 246,98 | probabilidad ≥ 0,720 | _Pendiente_ |
| PR-AUC | — | 0,8661 | _Pendiente_ |
| Costo de negocio (USD) | 179.037,29 | 87.051,60 (-51%) | _Pendiente_ |
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
completa se hace en GitHub Codespaces.

| Perfil de Docker Compose | Contenedor | Límite | Uso medido |
|---------------------------|-----------|--------|------------|
| `entrenamiento` | `mlflow` | 1536 MB | Se estabiliza contra el límite (`docker stats` incluye caché de página, no solo memoria real). Con 512-768 MB entraba en un loop de reinicios (`RestartCount` subiendo) al loguear un modelo LightGBM real; con los 4 workers por defecto de `mlflow server` la presión era aún mayor — se bajó a `--workers 1`. |
