"""Simple JSON-backed local disk cache, keyed by string."""

import json
import os

from config import CACHE_DIR


def _path(namespace: str, key: str) -> str:
    d = os.path.join(CACHE_DIR, namespace)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{key}.json")


def get(namespace: str, key: str):
    p = _path(namespace, key)
    if not os.path.exists(p):
        return None
    with open(p, "r") as f:
        return json.load(f)


def set(namespace: str, key: str, value) -> None:
    p = _path(namespace, key)
    with open(p, "w") as f:
        json.dump(value, f)
