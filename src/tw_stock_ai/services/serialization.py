from __future__ import annotations

from fastapi.encoders import jsonable_encoder


def to_jsonable(value):
    return jsonable_encoder(value)
