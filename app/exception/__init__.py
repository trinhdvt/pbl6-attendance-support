class CustomException(Exception):
    def __init__(self, message: str, status_code: int):
        self.message = message
        self.status_code = status_code

    def to_json(self) -> dict:
        return {
            'message': self.message,
            'status_code': self.status_code
        }
