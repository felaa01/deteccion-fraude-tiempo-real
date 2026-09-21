.PHONY: instalar lint formatear tipos pruebas verificar mlflow-arriba mlflow-abajo entrenar servicio-arriba servicio-abajo calcular-historico tiempo-real-arriba tiempo-real-abajo kafka-crear-topico productor arranque-en-frio streaming kafka-reiniciar-topico streaming-reiniciar skew-punta-a-punta

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

# Perfil "tiempo-real": Kafka + Redis (el job de Spark corre en el host con uv).
tiempo-real-arriba:
	docker compose --profile tiempo-real up -d

tiempo-real-abajo:
	docker compose --profile tiempo-real down

kafka-crear-topico:
	docker compose --profile tiempo-real exec kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 \
		--create --if-not-exists --topic transacciones --partitions 3 --replication-factor 1

# Reproduce fraudTest en Kafka. Ejemplo: make productor ARGS="--aceleracion 0 --limite 1000"
productor:
	uv run python -m fraude.productor.publicar $(ARGS)

# Estado por tarjeta al corte de fraudTrain: Spark lo calcula a Parquet y Feast lo materializa
# en Redis. Necesita Redis arriba (make tiempo-real-arriba).
arranque-en-frio:
	uv run python -m fraude.lotes.calcular_estado_inicial
	uv run python -m fraude.caracteristicas.arranque_en_frio

# Job de Spark Structured Streaming: Kafka -> estado por tarjeta en Redis. Corre en el host con
# uv. Ejemplo: make streaming ARGS="--hasta-agotar" (procesa lo que hay y termina).
streaming:
	uv run python -m fraude.tiempo_real.streaming_estado $(ARGS)

kafka-reiniciar-topico:
	docker compose --profile tiempo-real exec kafka /opt/kafka/bin/kafka-topics.sh \
		--bootstrap-server localhost:9092 --delete --topic transacciones --if-exists
	sleep 3
	$(MAKE) kafka-crear-topico

# Vuelve todo al punto de partida para una corrida limpia: tópico vacío, sin checkpoint (los
# offsets guardados no valen para un tópico nuevo) y el estado de Redis al corte de fraudTrain.
# Se vacía la base 0 de Redis antes de la carga: el arranque en frío solo escribe las tarjetas
# de fraudTrain y, sin vaciar, las que aparecen recién en fraudTest conservarían su estado viejo.
streaming-reiniciar: kafka-reiniciar-topico
	rm -rf datos_features/checkpoint_streaming
	docker compose --profile tiempo-real exec redis redis-cli -n 0 flushdb
	$(MAKE) arranque-en-frio

# Training-serving skew de punta a punta (regla 3): productor -> Kafka -> Spark -> Redis -> servicio
# contra el batch offline, con datos reales de fraudTest (~35 s). Necesita Kafka y Redis arriba
# (make tiempo-real-arriba) y los Parquet de make calcular-historico y make arranque-en-frio. Usa
# la base 1 de Redis, sin tocar el estado real de la base 0.
skew-punta-a-punta:
	uv run pytest -m integracion pruebas/prueba_skew_punta_a_punta.py --no-cov
