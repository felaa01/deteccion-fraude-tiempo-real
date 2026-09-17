# Plan del proyecto — Detección de fraude con tarjetas en tiempo real

## Datos

El "Credit Card Transactions Fraud Detection Dataset" generado con Sparkov, disponible en Kaggle
(alrededor de 1,8 millones de transacciones). A diferencia de IEEE-CIS, incluye número de tarjeta,
comercio, categoría, coordenadas del cliente y del comercio y marcas de tiempo, que son necesarias
para las características de velocidad y distancia. Las columnas se renombran al español al cargar.

Por la máquina de 8 GB, el dataset de Kaggle es la fuente principal. El generador Sparkov, que es
de código abierto, se usa solo en volúmenes chicos para inyectar un patrón de fraude nuevo en la
demo de drift. Spark sigue justificado por la parte de streaming.

## Arquitectura

### Streaming

- Un productor lee transacciones y las publica en un tópico de Kafka a velocidad acelerada.
- Spark Structured Streaming las consume y calcula características por tarjeta en ventanas:
  cantidad de transacciones en los últimos 10 minutos, 1 hora y 24 horas; monto acumulado;
  relación entre el monto actual y el promedio de la tarjeta; si la categoría del comercio es nueva
  para esa tarjeta.
- Los resultados se envían a Feast mediante un push source hacia Redis (almacén online).

### Características en el momento de la solicitud

Algunas características no pueden esperar al micro-batch de Spark: la distancia desde la
transacción anterior y la velocidad implícita (una tarjeta usada en Montevideo y 20 minutos después
en Madrid). Se calculan en el servicio de predicción a partir de la última ubicación guardada en
Redis. Documentar por qué se dividen así las características (frescura frente a latencia); es un
tema clave de entrevista.

### Offline

- PySpark calcula las mismas características sobre todo el histórico y las guarda en Parquet como
  almacén offline de Feast.
- Los conjuntos de entrenamiento se arman con joins point-in-time para evitar filtrar información
  del futuro.
- **Prueba de training-serving skew:** calcular las características de una muestra por la vía
  offline y por la vía online, y verificar que coincidan.

## Modelado

- **Partición temporal**, nunca aleatoria.
- **Tres modelos:** una línea base de reglas simples, gradient boosting (LightGBM o XGBoost) y un
  modelo en PyTorch con embeddings para las variables categóricas como retador.
- **Desbalance de clases** (el fraude es menos del 1%): pesos por clase o focal loss; nunca
  remuestrear el conjunto de prueba.
- **Métricas orientadas al negocio:** PR-AUC, recall con una tasa de falsos positivos fija y una
  función de costo: monto de fraude no detectado más un costo por cada cliente legítimo bloqueado.
  El umbral minimiza ese costo y se documenta como una decisión de negocio configurable.
- **MLflow** registra cada corrida con parámetros, métricas, umbral y artefactos; el registro de
  modelos usa los alias `campeon` y `retador`.

## Orquestación y etiquetas demoradas

Simular la demora de las etiquetas: si una transacción fue fraude recién se hace visible para el
sistema después de un tiempo configurable, como pasa con los contracargos reales. Esto afecta tanto
al reentrenamiento como al monitoreo.

DAG semanal de Airflow: calcula características con las etiquetas disponibles hasta ese momento,
arma el conjunto de entrenamiento desde Feast, entrena, evalúa sobre la ventana etiquetada más
reciente y promueve al retador solo si reduce el costo frente al campeón.

## Servicio de predicción y monitoreo

- **Servicio:** FastAPI en Docker sobre un clúster local de Kubernetes (kind), con manifiestos o
  Helm y autoescalado horizontal. Presupuesto de latencia, por ejemplo p99 menor a 50 ms,
  verificado con pruebas de carga en Locust o k6 (en Codespaces, no en la máquina de 8 GB).
- **Despliegue en sombra:** el retador puntúa cada transacción en paralelo; su resultado solo se
  registra y nunca afecta la decisión.
- **Monitoreo:** Prometheus y Grafana para latencia, distribución de puntajes y tasa de alertas.
  Evidently para el drift de características. El drift de predicciones se monitorea de inmediato;
  el rendimiento real, cuando llegan las etiquetas.

### La demo

Inyectar con el generador un patrón de fraude nuevo (por ejemplo, muchas compras online pequeñas en
tarjetas usadas recientemente). Mostrar que el modelo actual no lo detecta, que salta la alerta de
drift, que se dispara el reentrenamiento y que el nuevo modelo lo detecta. Grabar un video de dos
minutos.

## Estrategia de recursos (máquina de 8 GB)

| Dónde | Qué corre ahí |
|-------|---------------|
| Local, un perfil de Docker Compose por vez | Entrenamiento + MLflow + FastAPI; Kafka + Spark Streaming + Redis/Feast; Airflow; kind + servicio + Prometheus/Grafana |
| GitHub Codespaces (4 núcleos, 16 GB) | Integración completa, pruebas de carga, grabación de la demo |
| Notebook de Kaggle | Entrenamientos pesados puntuales, si hacen falta |

Plan B si Codespaces no alcanza: el nivel gratuito de Oracle Cloud (reducido a 2 núcleos ARM y 12 GB
en junio de 2026; las imágenes ARM y la aprobación de la cuenta pueden dar problemas).

## Estructura del repositorio

```
deteccion-fraude-tiempo-real/
├── src/fraude/
│   ├── productor/        # generador y productor de Kafka
│   ├── tiempo_real/      # jobs de Spark Structured Streaming
│   ├── lotes/            # cálculo histórico de características con PySpark
│   ├── caracteristicas/  # definiciones de Feast
│   ├── entrenamiento/    # modelos, métricas de costo, umbral
│   ├── servicio/         # API FastAPI, características en el momento de la solicitud
│   └── monitoreo/        # Evidently, métricas de Prometheus
├── dags/                 # Airflow
├── despliegue/k8s/       # manifiestos o Helm
├── pruebas_carga/
├── docs/                 # plan y configuración de la máquina
├── pruebas/              # incluye la prueba de skew
├── docker-compose.yml    # un perfil por subsistema
└── .github/workflows/
```

## Plan por semanas (7 semanas a tiempo completo)

- **Semana 1.** Datos, análisis exploratorio, partición temporal, línea base y gradient boosting con
  métricas de costo, MLflow.
- **Semana 2.** Modelo en PyTorch, selección de umbral, servicio FastAPI en Docker.
- **Semana 3.** Cálculo histórico de características con PySpark, definiciones de Feast con almacén
  offline y joins point-in-time.
- **Semana 4.** Kafka, Spark Structured Streaming, push a Redis, características en el momento de la
  solicitud y prueba de skew.
- **Semana 5.** DAG de Airflow con etiquetas demoradas y promoción campeón/retador.
- **Semana 6.** Kubernetes con autoescalado, despliegue en sombra y monitoreo.
- **Semana 7.** Entorno de integración en Codespaces, pruebas de carga, demo de drift, README y
  video.

## Métricas a reportar (para el CV y el README)

Reducción de costo frente a la línea base de reglas, PR-AUC, latencia p99 del servicio, tiempo de
reentrenamiento y memoria medida por componente.
