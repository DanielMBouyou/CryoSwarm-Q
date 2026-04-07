from __future__ import annotations

"""Simple event bus primitives for orchestration monitoring."""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from packages.core.logging import get_logger


logger = get_logger(__name__)

EventHandler = Callable[["PipelineEvent"], None]


@dataclass(frozen=True, slots=True)
class PipelineEvent:
    """An immutable orchestration event emitted during pipeline execution."""

    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class EventBus:
    """A tiny in-process observer bus with replayable event history."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[EventHandler]] = defaultdict(list)
        self._history: list[PipelineEvent] = []

    def subscribe(self, event_type: str, handler: EventHandler) -> Callable[[], None]:
        """Register a handler and return an unsubscribe callback."""

        self._subscribers[event_type].append(handler)

        def _unsubscribe() -> None:
            handlers = self._subscribers.get(event_type)
            if not handlers:
                return
            try:
                handlers.remove(handler)
            except ValueError:
                return

        return _unsubscribe

    def publish(self, event_type: str, **payload: Any) -> PipelineEvent:
        """Publish an event to specific and wildcard subscribers."""

        event = PipelineEvent(event_type=event_type, payload=dict(payload))
        self._history.append(event)
        handlers = [
            *self._subscribers.get(event_type, []),
            *self._subscribers.get("*", []),
        ]
        for handler in handlers:
            try:
                handler(event)
            except Exception as exc:
                logger.warning(
                    "Event handler for %s failed with %s: %s",
                    event_type,
                    type(exc).__name__,
                    exc,
                )
        return event

    @property
    def history(self) -> tuple[PipelineEvent, ...]:
        return tuple(self._history)
