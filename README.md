# Detección de fraude con tarjetas en tiempo real

Puntúa transacciones con tarjeta en milisegundos a medida que llegan, con características en
streaming, feature store, reentrenamiento automático con etiquetas demoradas y monitoreo de drift.

> En desarrollo.

## Arquitectura

_Diagrama: Kafka → Spark Structured Streaming → Feast (Redis) → FastAPI en Kubernetes, con
características históricas en PySpark, MLflow, reentrenamiento con Airflow y
Prometheus/Grafana/Evidently._

## Decisiones de diseño

_Se documentan a medida que avanza el proyecto._

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

## Presupuesto de recursos

Desarrollado en una máquina de 8 GB: cada subsistema corre por separado en local y la integración
completa se hace en GitHub Codespaces. Aquí se documentará la memoria medida de cada componente.
