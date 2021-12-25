import json


class TaskException(Exception):
    """
    Custom exception for celery task.
    """

    def __init__(self, message, status="FAILED"):
        self.message = message
        self.status = status

    def __str__(self):
        return json.dumps(self.to_json(), indent=4)

    def to_json(self):
        return {
            'message': self.message,
            'status': self.status
        }
