import json
import os
import time
import psycopg2
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

def get_db_conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        user=os.getenv("DB_USER", "eagle"),
        password=os.getenv("DB_PASS", "eaglepass"),
        dbname=os.getenv("DB_NAME", "equipmentdb")
    )

def create_consumer():
    broker = os.getenv("KAFKA_BROKER", "localhost:9092")
    while True:
        try:
            print(f"Connecting to Kafka at {broker}...")
            consumer = KafkaConsumer(
                "equipment-events",
                bootstrap_servers=broker,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="earliest",
                group_id="equipment-consumer-group",
                request_timeout_ms=30000,
                
            )
            print("Connected to Kafka successfully!")
            return consumer
        except NoBrokersAvailable:
            print("Kafka not ready, waiting 10s...")
            time.sleep(10)
        except Exception as e:
            print(f"Kafka error: {e}")
            time.sleep(10)

def main():
    print("Starting Consumer Service...")
    time.sleep(20)

    consumer = create_consumer()

    conn = None
    while conn is None:
        try:
            conn = get_db_conn()
            print("Connected to database!")
        except Exception as e:
            print(f"DB not ready: {e}")
            time.sleep(5)

    cursor = conn.cursor()
    print("Waiting for messages...")

    for message in consumer:
        try:
            data = message.value

            u = data["utilization"]
            t = data["time_analytics"]

            cursor.execute("""
                INSERT INTO equipment_events (
                    frame_id, equipment_id, equipment_class,
                    current_state, current_activity, motion_source,
                    total_tracked_seconds, total_active_seconds,
                    total_idle_seconds, utilization_percent
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                data["frame_id"],
                data["equipment_id"],
                data["equipment_class"],
                u["current_state"],
                u["current_activity"],
                u["motion_source"],
                t["total_tracked_seconds"],
                t["total_active_seconds"],
                t["total_idle_seconds"],
                t["utilization_percent"]
            ))
            conn.commit()
        

        except Exception as e:
            print(f"Error processing message: {e}")
            print(f"Message was: {message.value}")
            conn.rollback()

if __name__ == "__main__":
    main()