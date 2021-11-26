from fastapi import FastAPI, status
from fastapi.exceptions import RequestValidationError, FastAPIError
from fastapi.responses import JSONResponse

from .exception import CustomException


def add_exception_handler(app: FastAPI) -> None:
    # custom request validation error handler
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc):
        return JSONResponse(content={
            "message": "Invalid or missing parameters"
        }, status_code=status.HTTP_400_BAD_REQUEST)

    # custom exception handler
    @app.exception_handler(CustomException)
    async def custom_exception_handler(request, exc: CustomException):
        return JSONResponse(
            content=exc.to_json(),
            status_code=exc.status_code
        )

    # global exception handler
    @app.exception_handler(FastAPIError)
    async def global_exception_handler(request, exc: FastAPIError):
        return JSONResponse(content={
            'message': str(exc),
        }, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
