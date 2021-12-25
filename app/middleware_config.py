import os

from fastapi import FastAPI, status
from fastapi.requests import Request
from fastapi.responses import JSONResponse

from .exception import CustomException


def add_middleware(app: FastAPI) -> None:
    # add CORSMiddleware
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(CORSMiddleware,
                       allow_origins=['*'],
                       allow_credentials=True,
                       allow_methods=['*'],
                       allow_headers=['*'])

    # add TrustedHostMiddleware
    # from fastapi.middleware.trustedhost import TrustedHostMiddleware
    # app.add_middleware(TrustedHostMiddleware,
    #                    allowed_hosts=['*.illusion.codes', 'illusion.codes'])

    # add GZipMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    app.add_middleware(GZipMiddleware, minimum_size=100 * 1024)  # 100KB

    # add max request size check Middleware
    @app.middleware("http")
    async def max_request_size_middleware(request: Request, call_next):
        #
        MAX_REQUEST_SIZE = int(os.getenv("MAX_REQUEST_SIZE", 1024 * 1024 * 1))
        content_length = int(request.headers.get('Content-Length', 0))

        #
        if content_length > MAX_REQUEST_SIZE:
            # response with 413 Error
            exception = CustomException(message="Request too large",
                                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)
            return JSONResponse(content=exception.to_json(),
                                status_code=exception.status_code)

        return await call_next(request)
