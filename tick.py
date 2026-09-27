"""Tick dispatch — decide what (if anything) to proactively send.

Placeholder for phase 0. Phase 3 fills in real signal selection + composition.
"""

from __future__ import annotations

from typing import Any

from store import DispatchState


def build_actions(available_triggers: list, now: str) -> list[dict[str, Any]]:
    return []
