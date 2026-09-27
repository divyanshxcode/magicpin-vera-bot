"""Process entrypoint.

Render runs `python main.py`; Railway runs the uvicorn startCommand in
railway.toml. Both reach this same app, and both need the bind address to be
0.0.0.0.

Binding to 0.0.0.0 is not optional. Uvicorn defaults to 127.0.0.1, which is
unreachable from outside the container, and the judge's healthcheck would time
out against a service that is otherwise working perfectly.

PORT comes from the platform. Render and Railway both inject it; the 8080
fallback is only for running this by hand.
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
