.PHONY: instalar lint formatear tipos pruebas verificar mlflow-arriba mlflow-abajo entrenar servicio-arriba servicio-abajo

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

servicio-arriba:
	docker compose --profile servicio up -d --build

servicio-abajo:
	docker compose --profile servicio down
