"""Process entrypoint.

Railway runs `python -m server` (or the uvicorn startCommand in railway.toml);
this makes the app importable and runnable either way, and keeps the bind
address correct for a container.

Binding to 0.0.0.0 is not optional. Uvicorn defaults to 127.0.0.1, which is
unreachable from outside the container, and the judge's healthcheck would time
out against a service that is otherwise working perfectly.
"""

from __future__ import annotations

import os

import uvicorn

from config import BOT_VERSION
from server import app  # noqa: F401  (re-exported for `uvicorn server:app`)

__all__ = ["app", "main"]


def main() -> None:
    uvicorn.run(
        "server:app",
        host="0.0.0.0",  # noqa: S104 - required for container/railway
        port=int(os.environ.get("PORT", "8080")),
        log_level=os.environ.get("LOG_LEVEL", "info"),
        access_log=False,
    )


if __name__ == "__main__":
    main()
