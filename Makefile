.PHONY: instalar lint formatear tipos pruebas verificar mlflow-arriba mlflow-abajo entrenar servicio-arriba servicio-abajo calcular-historico kafka-arriba kafka-abajo kafka-crear-topico productor

instalar:
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check .

formatear:
	uv run ruff format .

tipos:
	uv run mypy

pruebas:
	uv run pytest -m "not integracion"

verificar: lint tipos pruebas
	uv run ruff format --check .

mlflow-arriba:
	docker compose --profile entrenamiento up -d

mlflow-abajo:
	docker compose --profile entrenamiento down

entrenar:
	uv run python -m fraude.entrenamiento.entrenar

calcular-historico:
	uv run python -m fraude.lotes.calcular_historico

servicio-arriba:
	docker compose --profile servicio up -d --build

servicio-abajo:
	docker compose --profile servicio down

kafka-arriba:
	docker compose --profile tiempo-real up -d kafka

kafka-abajo:
	docker compose --profile tiempo-real down

kafka-crear-topico:
	docker compose --profile tiempo-real exec kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 \
		--create --if-not-exists --topic transacciones --partitions 3 --replication-factor 1

# Reproduce fraudTest en Kafka. Ejemplo: make productor ARGS="--aceleracion 0 --limite 1000"
productor:
	uv run python -m fraude.productor.publicar $(ARGS)
