import json
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
import os
import time

def create_producer():
    broker = os.getenv("KAFKA_BROKER", "localhost:9092")

    # Infinite retry loop — keeps trying until Kafka is ready
    # This is necessary because Kafka takes time to fully start
    # even after its container is marked as running
    attempt = 0
    while True:
        try:
            attempt += 1
            print(f"Connecting to Kafka at {broker} (attempt {attempt})...")

            producer = KafkaProducer(
                bootstrap_servers=broker,

                # Serialize Python dict to JSON bytes before sending
                # Kafka only transports raw bytes, not Python objects
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),

                # How long to wait for a response from the broker (ms)
                request_timeout_ms=30000,

                # How long to wait when fetching metadata from broker (ms)
                max_block_ms=30000,
            )
            print("Connected to Kafka successfully!")
            return producer

        except NoBrokersAvailable:
            # Kafka port is open but broker isn't ready yet
            print(f"Kafka not ready, waiting 10s before retry...")
            time.sleep(10)

        except Exception as e:
            # Any other unexpected error
            print(f"Unexpected error connecting to Kafka: {e}")
            time.sleep(10)

def send_payload(producer, payload: dict):
    # Send payload to the "equipment-events" topic
    # A topic is like a named channel/queue in Kafka
    producer.send("equipment-events", value=payload)

    # Force immediate send — without this Kafka might batch
    # messages and delay delivery
    producer.flush()