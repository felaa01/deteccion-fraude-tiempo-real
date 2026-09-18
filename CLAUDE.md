# CLAUDE.md — Detección de fraude con tarjetas en tiempo real

## Para qué existe este proyecto

Proyecto de portfolio para demostrar las habilidades que piden los puestos de **Machine Learning
Engineer**: diseñar, entrenar y desplegar modelos de ML en producción, prácticas de MLOps, Python
con PyTorch, Docker y Kubernetes, orquestación de ML (MLflow, Airflow), feature stores (Feast),
herramientas de datos a gran escala (Spark, Kafka), análisis de investigación y comunicación.

La especificación completa y el plan por semanas están en @docs/plan-del-proyecto.md.

## Cómo trabajar conmigo

- **Respondeme siempre en español rioplatense (con "vos").**
- **Todo el proyecto va en español:** código (nombres de variables, funciones, clases, módulos y
  carpetas), comentarios, docstrings, mensajes de commit, pull requests, documentación, README,
  nombres de características, métricas y alias de modelos.
  - Identificadores en español **sin tildes ni ñ** (por ejemplo `monto_promedio_tarjeta`,
    `distancia_transaccion_anterior`). Tildes y ñ sí van en comentarios, docstrings y documentación.
  - Excepciones inevitables: palabras reservadas de Python, nombres de librerías y sus APIs,
    columnas originales del dataset (se renombran al cargarlo), archivos que las herramientas
    exigen con ese nombre (`CLAUDE.md`, `README.md`, `Makefile`, `pyproject.toml`, `uv.lock`,
    `docker-compose.yml`, `Dockerfile`, `.github/workflows/`, `.pre-commit-config.yaml`, `dags/`)
    y términos técnicos sin traducción de uso común (streaming, pipeline, feature store, drift).
- Voy a tener que explicar y defender cada decisión de diseño en entrevistas técnicas. Antes de
  implementar algo no trivial, explicame brevemente el enfoque y los trade-offs, y consultame
  cuando haya una decisión de diseño real. Preferí pasos chicos y revisables en lugar de mucho
  código de golpe, y decime qué leer para entender lo que construiste.
- Si un requisito es ambiguo, preguntá en lugar de suponer.
- Las APIs de las librerías cambian: revisá la documentación actual o la versión instalada antes
  de usar una API de memoria (sobre todo Feast, Airflow, Spark, Evidently y MLflow). Usá las
  versiones estables más recientes de cada herramienta.

## Restricciones obligatorias

- **Costo cero.** Nunca agregues un servicio pago ni nada que pueda generar un cobro sin
  consultarme antes. Todos los componentes son de código abierto y corren en local o en entornos
  gratuitos.
- **Máquina:** Windows 11 + WSL2 (Ubuntu 24.04), **8 GB de RAM en total, WSL limitado a 5 GB,
  sin GPU**. Docker Engine corre dentro de WSL (sin Docker Desktop). El repositorio está en
  `~/proyectos`, nunca en `/mnt/c`. Los archivos de configuración de la máquina están en
  `docs/configuracion/`.
- **El sistema completo no entra en esta máquina. Nunca diseñes un flujo de trabajo que necesite
  Kafka, Spark, Airflow, Redis, MLflow, Kubernetes y el monitoreo corriendo al mismo tiempo en
  local.**
  - El desarrollo local se hace **de a un subsistema por vez**, cada uno dentro de 3 a 4 GB
    aproximadamente, usando **perfiles de Docker Compose** (por ejemplo
    `docker compose --profile tiempo-real up`).
  - Cada contenedor lleva un límite de memoria explícito. Medir el uso real y registrarlo en la
    sección de presupuesto de recursos del README.
  - Kafka en modo KRaft (sin ZooKeeper). Spark en modo local con poca memoria para el driver.
    Airflow con configuración mínima. kind con un solo nodo.
  - **La integración completa de punta a punta, las pruebas de carga y la demo se hacen en GitHub
    Codespaces** (máquina de 4 núcleos y 16 GB; la cuota gratuita es de 120 core-hours por mes,
    unas 30 horas en esa máquina). Codespaces es para integrar, no para el desarrollo diario.
    Detener el codespace al terminar y vigilar la cuota de 15 GB de almacenamiento (las imágenes de
    Spark y Airflow son grandes).
  - Los trabajos pesados puntuales (entrenamientos grandes) pueden correr en una notebook gratuita
    de Kaggle.
- Spark necesita Java 17 o superior en WSL (`openjdk-17-jre-headless`). Verificar la
  compatibilidad con Python 3.12 de cada dependencia importante antes de agregarla.
- **Airflow corre en su propio contenedor**, no dentro del entorno virtual del proyecto, para
  evitar conflictos de dependencias. Los DAGs llaman al código del proyecto mediante contenedores o
  una interfaz mínima.

## Comandos

```bash
make instalar    # uv sync + hooks de pre-commit
make verificar   # lint + formato + mypy + pruebas (lo mismo que el CI)
make pruebas     # pytest, sin las pruebas marcadas "integracion"
```

Las dependencias se gestionan con **uv** (`uv add <paquete>`, `uv add --dev <paquete>`). `uv.lock`
se versiona. Agregar objetivos del Makefile en español para cada perfil de Docker Compose a medida
que se construyen los subsistemas.

## Convenciones de trabajo

- Python 3.12, anotaciones de tipos completas, **mypy en modo estricto** tiene que pasar. ruff para
  lint y formato.
- pytest para todo. Las pruebas viven en `pruebas/`, en archivos `prueba_*.py` con funciones
  `prueba_*` (ya configurado en `pyproject.toml`). Las pruebas que necesitan servicios en ejecución
  (Kafka, Redis, etc.) se marcan con `@pytest.mark.integracion` y quedan fuera del CI; las pruebas
  unitarias usan datos pequeños en memoria.
- Trabajar en ramas e integrar mediante pull requests, aunque trabaje solo.
- Commits chicos con mensajes claros en español. Nunca versionar datos, artefactos de modelos,
  `mlruns/` ni `.env`.
- Desarrollar con muestras de datos; usar el dataset completo solo cuando el pipeline funcione.
- Mantener el README actualizado con arquitectura, decisiones, resultados y presupuesto de recursos.

## Reglas de ML no negociables

1. **Solo partición temporal**, nunca aleatoria: entrenar con meses anteriores y validar y probar
   con los posteriores.
2. **Joins point-in-time correctos** para armar los datos de entrenamiento. Ninguna característica
   puede usar información posterior al momento de la transacción.
3. **Las características offline y online tienen que coincidir.** Una prueba de training-serving
   skew calcula las características de una muestra por las dos vías y verifica que sean iguales.
4. **Nunca remuestrear el conjunto de prueba.** El desbalance se maneja con pesos por clase o
   focal loss.
5. **El accuracy no sirve acá.** Usar PR-AUC, recall con una tasa de falsos positivos fija y una
   función de costo de negocio; el umbral de decisión minimiza ese costo y es configurable.
6. **Las etiquetas llegan con demora** (los contracargos tardan días o semanas). El reentrenamiento
   y el monitoreo de rendimiento solo pueden usar etiquetas que habrían estado disponibles en ese
   momento.
7. Un modelo retador solo se promueve si reduce el costo de negocio frente al campeón en la ventana
   etiquetada más reciente. Los alias del registro de MLflow son `campeon` y `retador`.

## Estado actual

Hecho:
- Esqueleto del repositorio con ruff, mypy estricto, pytest con cobertura, pre-commit y CI en
  GitHub Actions.
- Estructura de paquetes en `src/fraude/` según el plan, con una prueba mínima.
- Repo movido a WSL2 (`~/proyectos/deteccion-fraude-tiempo-real`), git inicializado.
- Dataset de Kaggle (kartik2112/fraud-detection, Sparkov) descargado en `datos/`.
- Carga y renombre de columnas al español (`src/fraude/entrenamiento/carga.py`) y partición
  temporal en tres partes: entrenamiento/validación/prueba (`src/fraude/entrenamiento/division.py`),
  con pruebas unitarias.
- Primer notebook de análisis exploratorio (`notebooks/01_analisis_exploratorio.py`, jupytext).
- Métrica de costo de negocio (`entrenamiento/costo.py`) y características básicas "sin historia"
  (`entrenamiento/caracteristicas_basicas.py`).
- Línea base de reglas por monto (`entrenamiento/linea_base.py`) y modelo LightGBM
  (`entrenamiento/modelo.py`), con selección de umbral por costo en validación.
- MLflow en Docker (perfil `entrenamiento`, `make mlflow-arriba` / `make mlflow-abajo`), con
  artefactos servidos vía proxy (`mlflow-artifacts:/`). Límite de memoria: 1536 MB (ver
  presupuesto de recursos en el README).
- Script de orquestación `entrenamiento/entrenar.py` (`make entrenar`) corrido contra el dataset
  completo: línea base costo USD 179.037 vs LightGBM costo USD 87.052 (-51%), PR-AUC 0,8661.
- Repo remoto en GitHub (público): https://github.com/felaa01/deteccion-fraude-tiempo-real.
  Semana 1 mergeada a `main` vía PR #1. Flujo validado: rama por hito → PR → merge.
- Modelo retador en PyTorch (`entrenamiento/modelo_pytorch.py`): MLP con embeddings para
  `categoria` y `genero`, `pos_weight` en `BCEWithLogitsLoss` para el desbalance (nunca
  remuestrear), selección de época por costo en validación. Dependencia CPU-only
  (`tool.uv.sources` en `pyproject.toml`, evita las ruedas CUDA de PyPI). Corrido contra el
  dataset completo: costo USD 58.469 vs LightGBM USD 87.052 (-33%), PR-AUC 0,7867 (menor que
  LightGBM, pero manda el costo, no el AUC). Registered model renombrado de
  `deteccion-fraude-lightgbm` a `deteccion-fraude` (nombre neutral para alojar ambas
  arquitecturas bajo los mismos alias).
- Corregido un bug real antes de mergear: `_correr_lightgbm` promovía su propia versión a
  `campeon` sin condición en cada corrida, lo que iba a revertir en silencio cualquier
  promoción manual del retador. Ahora ninguna corrida mueve el alias solo: se compara contra
  el costo ya logueado de la versión que hoy tiene `campeon` (con fallback a infinito si
  todavía no existe ninguna), y la promoción sigue siendo manual para las dos arquitecturas.
- Wrapper pyfunc de MLflow (`entrenamiento/envoltorio_pyfunc.py`): LightGBM y PyTorch se
  loguean con la misma interfaz `mlflow.pyfunc.PythonModel` (cada uno envolviendo su propia
  lógica de predicción), para que el servicio siempre pueda hacer
  `mlflow.pyfunc.load_model(...).predict(df)` sin importar la arquitectura detrás del alias
  `campeon`. La versión 3 (PyTorch, logueada antes de este cambio con el flavor nativo) quedó
  incompatible con la carga genérica — se repaqueteó como versión 7 (mismo código, misma
  semilla, mismo costo) y se movió `campeon` ahí; fue una migración técnica, no una promoción
  por costo.
- Servicio FastAPI (`servicio/principal.py`, `servicio/modelo_actual.py`,
  `servicio/esquemas.py`): `POST /predecir` calcula las características con la misma función
  del entrenamiento (`agregar_caracteristicas_basicas`) y aplica el umbral logueado en el run
  del campeón. Dockerizado (`despliegue/servicio/Dockerfile`, perfil `servicio` con `mlflow` +
  `api`, `make servicio-arriba` / `make servicio-abajo`). MLflow necesitó
  `--allowed-hosts localhost,mlflow:5000` porque rechaza por defecto el header `Host: mlflow`
  que le llega desde el contenedor `api` (protección contra DNS rebinding). Validado de punta
  a punta contra el dataset completo y contra los contenedores reales, no solo con mocks.
  Memoria medida: `api` ~292 MB en reposo (límite 1024 MB).
- Cálculo histórico de características con PySpark
  (`lotes/caracteristicas_historicas.py`, `make calcular-historico`), sobre `fraudTrain` +
  `fraudTest` concatenados: conteos de transacciones en 10 min/1 h/24 h por tarjeta, monto
  acumulado, ratio contra el promedio histórico y bandera de categoría nueva. Point-in-time por
  diseño: las ventanas excluyen siempre la fila actual, así el valor guardado para la
  transacción T ya es lo que un servicio en producción sabría justo antes de T. Definiciones de
  Feast (`src/fraude/caracteristicas/`) con almacén offline Parquet (`type: file`) y `RepoConfig`
  armado en código en vez de `feature_store.yaml`; función de join point-in-time para armar
  datasets de entrenamiento (`caracteristicas/dataset_entrenamiento.py`). Agregadas las
  dependencias `pyspark` y `feast`; hubo que bajar `pandas` a `<3` porque Feast (incluso 0.66, la
  última) todavía no lo soporta -- sin ese tope, `uv` resolvía en silencio una versión de Feast de
  2022 (0.20.0). Bug real encontrado validando el join contra el dataset completo (no solo con
  datos sintéticos): `to_timestamp` de Spark usa la zona horaria de *sesión* si no se fija a UTC
  explícitamente, lo que en esta máquina (UTC-3) corría los timestamps del Parquet 3 horas y
  rompía en silencio el join de Feast -- no se notaba dentro de una misma sesión de Spark, porque
  `to_timestamp` seguido de `toPandas()` se cancela solo. Prueba de regresión agregada
  (`pruebas/prueba_calcular_historico.py`).

- Semana 4, en curso (rama `semana-4/kafka-streaming`): broker de Kafka 4.3.1 en KRaft (perfil
  `tiempo-real`, `make kafka-arriba`, tópico `transacciones` con 3 particiones, ~394 MB) y productor
  (`src/fraude/productor/`, `make productor`) con `confluent-kafka`: reproduce `fraudTest` ordenado por
  fecha (desempate por `id_transaccion`), clave = `numero_tarjeta`, ritmo acelerado agendado contra el
  inicio. El mensaje excluye `es_fraude` y datos personales. `unix_time` del dataset está desfasado 7
  años respecto de la fecha (2013 vs 2020): el event time es `fecha_hora_transaccion`. Validado contra
  el broker real: 0 tarjetas en más de una partición, 0 transacciones fuera de orden.

  Estado por tarjeta (`caracteristicas/estado_tarjeta.py`): funciones puras `actualizar_estado`
  (streaming, idempotente) y `calcular_caracteristicas` (momento de la solicitud), sin Spark/Redis/Feast.
  Redis guarda el estado crudo (marcas y montos de las últimas 24 h, acumulado, categorías), no los
  conteos ya calculados, porque estos dependen de la hora de la solicitud. Decisiones: el estado vive
  solo en Redis (Spark lo lee y lo actualiza en `foreachBatch`), Feast como capa de acceso y el arranque
  en frío materializa el estado al final de `fraudTrain`. Prueba de skew unitaria contra el batch de
  Spark (600 transacciones con empates de segundo y huecos de más de 24 h): pasa.

Próximos pasos (resto de la semana 4, según el plan):
1. Spark Structured Streaming, push a Redis (almacén online de Feast),
   características en el momento de la solicitud en el servicio, y la prueba de
   training-serving skew (offline vs. online).

Actualizá esta sección cada vez que se complete un hito.
