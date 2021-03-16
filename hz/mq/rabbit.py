import pika

from hz.mq.core import MQ


class RabbitMQ(MQ):
    def __init__(self, connection: str):
        super(RabbitMQ, self).__init__()

        self._channel = pika.BlockingConnection(pika.URLParameters(connection)).channel()
        self._channel.queue_declare('RedTown-RPC', durable=True, exclusive=False, auto_delete=False, arguments=None)

    def send(self, message):
        self._channel.basic_publish(exchange='', routing_key='H0-MQ', body=str(message).encode('utf-8'))
