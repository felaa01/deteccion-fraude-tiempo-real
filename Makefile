.PHONY: instalar lint formatear tipos pruebas verificar

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
