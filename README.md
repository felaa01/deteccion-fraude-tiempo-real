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
- **PyTorch como retador:** red con embeddings para `categoria` y `genero` (variables numéricas
  estandarizadas con media/desvío de train) y un MLP chico encima. El desbalance se maneja con
  `pos_weight` en `BCEWithLogitsLoss` (el equivalente de PyTorch al `class_weight="balanced"` de
  LightGBM), nunca remuestreando. La selección de época usa el costo de negocio en validación, igual
  que el umbral. Rueda CPU-only de PyTorch (`tool.uv.sources` en `pyproject.toml`): la rueda de PyPI
  trae dependencias CUDA de varios GB que no sirven sin GPU.
- **Un solo registered model (`deteccion-fraude`) para las dos arquitecturas:** los alias
  `campeon`/`retador` de MLflow apuntan a versiones dentro de un mismo modelo registrado, así que
  hace falta un nombre neutral (no `-lightgbm`) para poder comparar campeón y retador con esos
  alias sin importar el framework. La promoción de `campeon` es un paso manual (regla de negocio:
  un retador solo se promueve si reduce el costo frente al campeón vigente en el registro), no
  algo que decida solo el script de entrenamiento.
- **Wrapper pyfunc de MLflow (`entrenamiento/envoltorio_pyfunc.py`):** LightGBM y PyTorch se
  loguean con la misma interfaz `mlflow.pyfunc.PythonModel`, cada uno envolviendo su propia lógica
  de `predict_proba`/sigmoide + preprocesador. Así el servicio de predicción siempre hace
  `mlflow.pyfunc.load_model("models:/deteccion-fraude@campeon").predict(df)` sin necesitar saber
  qué arquitectura hay detrás del alias — la alternativa (que el servicio inspeccione el flavor y
  tenga una rama de carga por arquitectura) hubiera acoplado el serving a los detalles internos de
  cada modelo.
- **`pandas<3` por Feast:** Feast (incluso la última versión, 0.66) todavía tope a `pandas<3`. Sin
  ese tope en `pyproject.toml`, `uv` resolvía en silencio una versión de Feast de 2022 (0.20.0) para
  poder convivir con `pandas>=3` de la semana 1, en vez de fallar fuerte — con años de API vieja y
  un downgrade de `protobuf` de paso. Se bajó `pandas` a `<3` para poder usar el Feast actual; no hay
  código que dependa de una API específica de pandas 3.x.
- **Características históricas point-in-time excluyendo la fila actual
  (`src/fraude/lotes/caracteristicas_historicas.py`):** las ventanas de Spark (conteos en 10 min/1
  h/24 h, monto acumulado, promedio) miran solo transacciones *estrictamente anteriores* de la
  misma tarjeta, nunca la fila que se está procesando. Así, el valor guardado para la transacción T
  ya es "lo que un servicio en producción sabría justo antes de T", y el join point-in-time de
  Feast (que matchea por `event_timestamp <= tiempo de la entidad`, inclusive) cae solo en la
  propia fila de T sin necesitar un desfasaje adicional en el join.
- **El histórico se calcula sobre `fraudTrain` + `fraudTest` concatenados:** la historia de cada
  tarjeta es continua más allá del corte de train/test que se usa para el modelado; cortarla ahí
  arrancaría la ventana de cada tarjeta a mitad de su historia real. No hay fuga de información
  porque cada fila solo mira hacia atrás, dentro de su propia tarjeta.
- **`RepoConfig` de Feast armado en código, sin `feature_store.yaml`**
  (`src/fraude/caracteristicas/tienda.py`): no hay ninguna razón para depender de un archivo con
  rutas relativas cuando el proyecto ya sabe, en Python, dónde viven el registro y los datos.
  Almacén offline `type: file` (Parquet); almacén online sqlite como placeholder porque Feast lo
  exige igual para aplicar las definiciones, aunque todavía no se usa (llega en la semana 4, con el
  push a Redis desde Spark Streaming).
- **Bug real encontrado validando contra el dataset completo, no con datos sintéticos:**
  `to_timestamp` en Spark interpreta el string con la zona horaria de *sesión* (por default, la del
  sistema — UTC-3 en esta máquina). Sin fijarla a `UTC` explícitamente, el Parquet quedaba con todos
  los timestamps corridos 3 horas respecto del string original. El join point-in-time de Feast
  (armado con pandas, que no aplica ese corrimiento) dejaba de encontrar coincidencias exactas y
  perdía filas en silencio — sin ningún error. Se manifestaba solo al cruzar la frontera
  Spark→Parquet→pandas/Feast, nunca dentro de una misma sesión de Spark (`to_timestamp` seguido de
  `toPandas()` se cancela solo, aunque la zona horaria esté mal). Prueba de regresión en
  `pruebas/prueba_calcular_historico.py`, que escribe a Parquet real y relee con pandas en vez de
  comparar contra el propio `toPandas()` de Spark.

## Resultados

Evaluado sobre `fraudTest.csv` (conjunto de prueba final, nunca usado para elegir el umbral),
con la matriz de costo de [Decisiones de diseño](#decisiones-de-diseño):

| Métrica | Reglas (línea base) | Gradient boosting (LightGBM) | PyTorch (campeón) |
|---------|---------------------|-------------------------------|---------|
| Umbral elegido en validación | monto ≥ USD 246,98 | probabilidad ≥ 0,720 | probabilidad ≥ 0,930 |
| PR-AUC | — | 0,8661 | 0,7867 |
| Costo de negocio (USD) | 179.037,29 | 87.051,60 (-51% vs reglas) | 58.469,43 (-33% vs LightGBM) |
| Latencia p99 | — | _Pendiente_ | _Pendiente_ |

El retador de PyTorch tiene menor PR-AUC que LightGBM pero menor costo de negocio: eligió un
umbral más conservador (0,930) que evita revisiones innecesarias sin perder tanto en los fraudes
de mayor monto. Regla del proyecto: el costo de negocio manda, no el accuracy ni el AUC. Se
promovió a `campeon` en el registro de MLflow.

> El campeón quedó en la versión 7, no la 3: la versión 3 se logueó con el flavor nativo de
> PyTorch de MLflow, que asume un único tensor de entrada y no sabe nada de nuestro preprocesador
> ni de las columnas categóricas — rompía al cargarlo genéricamente con
> `mlflow.pyfunc.load_model`. La versión 7 es el mismo modelo (mismo código, misma semilla, mismo
> costo de prueba) re-entrenado y logueado con el wrapper pyfunc nuevo. Fue una migración técnica
> para poder servirlo, no una promoción por costo.

## Servicio de predicción (FastAPI)

`src/fraude/servicio/` expone el campeón vigente como una API HTTP:

- `GET /salud`: chequeo de vida.
- `POST /predecir`: recibe los campos crudos de una transacción (los mismos que necesita
  `agregar_caracteristicas_basicas`, para no calcular la característica dos veces con lógica
  distinta) y devuelve `probabilidad_fraude`, `es_fraude` (según el umbral que se eligió al
  entrenar el campeón) y la versión del modelo que respondió.

El modelo se carga una sola vez, al arrancar el proceso (`fraude.servicio.modelo_actual`); un
campeón nuevo se toma recién en el próximo reinicio del servicio — el refresco en caliente es un
problema de despliegue (semana 6, Kubernetes), no de esta API.

```bash
make servicio-arriba   # build + docker compose --profile servicio up -d (mlflow + api)
curl -X POST localhost:8000/predecir -H 'Content-Type: application/json' -d '{...}'
make servicio-abajo
```

**MLflow con `--allowed-hosts`:** el contenedor `api` le habla a MLflow como `http://mlflow:5000`
(el nombre del servicio en la red de Docker, no `localhost`). Las versiones recientes de MLflow
rechazan por defecto cualquier header `Host` que no sea `localhost` o una IP privada, como
protección contra ataques de DNS rebinding — hubo que agregar `--allowed-hosts localhost,mlflow:5000`
al comando del servidor para permitir explícitamente ese nombre.

## Flujo en tiempo real (Kafka)

Perfil `tiempo-real` de Docker Compose. Se levanta de a un subsistema por vez (no junto con MLflow
ni con el servicio).

```bash
make kafka-arriba         # broker de Kafka en modo KRaft, un solo nodo
make kafka-crear-topico   # tópico `transacciones`, 3 particiones
make productor ARGS="--limite 2000"   # reproduce fraudTest acelerado (x3600 por defecto)
make kafka-abajo
```

- **Clave del mensaje = `numero_tarjeta`.** Kafka solo garantiza orden dentro de una partición, y la
  partición sale del hash de la clave: todas las transacciones de una tarjeta se consumen en el
  orden en que ocurrieron, que es lo que necesitan las ventanas por tarjeta.
- **El mensaje no lleva `es_fraude`** (las etiquetas llegan con demora, regla 6) ni datos personales
  que ninguna característica use.
- **El tiempo del evento es `fecha_hora_transaccion`.** La columna `unix_time` del dataset está
  desfasada exactamente 7 años (2013 contra 2020) y no se usa.
- **Ritmo acelerado** (`--aceleracion`, segundos simulados por segundo real; 0 = sin esperas). Cada
  mensaje se agenda contra el inicio de la reproducción, no como "esperar la diferencia con el
  anterior", para que el error de cada `sleep` no se acumule. Verificado: 41.768 s simulados a x3600
  tardaron 11,6 s reales.
- Publicar dos veces sin recrear el tópico duplica los mensajes.

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
| `servicio` | `mlflow` | 1536 MB | Igual que en `entrenamiento`: solo hace falta para que la API cargue el campeón al arrancar. |
| `servicio` | `api` | 1024 MB | ~292 MB en reposo (`docker stats`), con PyTorch y LightGBM cargados en memoria. Queda margen: no se ajustó a la baja para no arriesgar OOM cuando lleguen ráfagas de pedidos concurrentes. |
| `tiempo-real` | `kafka` (KRaft, un nodo) | 1024 MB (heap JVM `-Xmx512m`) | ~394 MB en reposo con el tópico `transacciones` creado (`docker stats`). |
| _(sin Docker, script directo)_ | `make calcular-historico` (JVM de Spark, `local[4]`) | `spark.driver.memory=2g` | ~415 MB de RSS medidos a mitad de corrida (`ps`), lejos del límite de 2 GB. Corre en ~16-20 s sobre 1.852.394 filas. `local[4]`, no `local[*]`: no hace falta acaparar los 12 núcleos de la máquina para un dataset de este tamaño. |
