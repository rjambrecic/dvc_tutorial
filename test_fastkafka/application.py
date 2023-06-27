# content of the "application.py" file

from pydantic import BaseModel, Field, NonNegativeFloat

from fastkafka import FastKafka
from fastkafka._components.logger import get_logger

logger = get_logger(__name__)

class Data(BaseModel):
    data: NonNegativeFloat = Field(
        ..., example=0.5, description="Float data example"
    )

kafka_brokers = {
    "localhost": {
        "url": "localhost",
        "description": "local development kafka broker",
        "port": 9092,
    },
    "production": {
        "url": "kafka.airt.ai",
        "description": "production kafka broker",
        "port": 9092,
        "protocol": "kafka-secure",
        "security": {"type": "plain"},
    },
}

kafka_app = FastKafka(
    title="Demo Kafka app",
    kafka_brokers=kafka_brokers,
)

@kafka_app.consumes(topic="input_data", auto_offset_reset="latest")
async def on_input_data(msg: Data):
    logger.info(f"Got data: {msg.data}")
    await to_output_data(msg.data)


@kafka_app.produces(topic="output_data")
async def to_output_data(data: float) -> Data:
    processed_data = Data(data=data+1.0)
    return processed_data
