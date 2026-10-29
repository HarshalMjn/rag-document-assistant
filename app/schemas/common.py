from pydantic import BaseModel


class DataResponse[T](BaseModel):
    """Envelope for every successful API response: {"data": ...}."""

    data: T
