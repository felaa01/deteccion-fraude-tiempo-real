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

Próximos pasos (semana 1):
1. Verificar con el usuario que el entorno de `docs/configuracion/` esté instalado.
2. Descargar de Kaggle el "Credit Card Transactions Fraud Detection Dataset" (Sparkov) en `datos/`
   (no se versiona), renombrar las columnas al español y hacer el análisis exploratorio.
3. Partición temporal, línea base de reglas y modelo de gradient boosting, métrica de costo de
   negocio y selección de umbral, todo registrado en MLflow (corriendo en local en un contenedor
   con límite de memoria).

Actualizá esta sección cada vez que se complete un hito.
