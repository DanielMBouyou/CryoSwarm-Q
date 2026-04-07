from __future__ import annotations

"""WebSocket event fanout for live campaign monitoring."""

import asyncio
from collections import defaultdict
from typing import DefaultDict

from packages.orchestration.events import PipelineEvent


class CampaignEventBroadcaster:
    """Fan out pipeline events to WebSocket subscribers."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._subscribers: DefaultDict[str | None, set[asyncio.Queue[PipelineEvent]]] = defaultdict(set)

    def subscribe(self, campaign_id: str | None = None) -> asyncio.Queue[PipelineEvent]:
        queue: asyncio.Queue[PipelineEvent] = asyncio.Queue(maxsize=128)
        self._subscribers[campaign_id].add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[PipelineEvent], campaign_id: str | None = None) -> None:
        subscribers = self._subscribers.get(campaign_id)
        if not subscribers:
            return
        subscribers.discard(queue)
        if not subscribers:
            self._subscribers.pop(campaign_id, None)

    def publish(self, event: PipelineEvent) -> None:
        self._loop.call_soon_threadsafe(self._fanout, event)

    def _fanout(self, event: PipelineEvent) -> None:
        campaign_id = event.payload.get("campaign_id")
        queues = set(self._subscribers.get(None, set()))
        if campaign_id is not None:
            queues.update(self._subscribers.get(str(campaign_id), set()))

        for queue in queues:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                continue
