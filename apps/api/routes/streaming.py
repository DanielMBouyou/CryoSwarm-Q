from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from apps.api.auth import verify_websocket_api_key


router = APIRouter(tags=["streaming"])


@router.websocket("/ws/campaigns")
async def campaign_progress_stream(
    websocket: WebSocket,
    campaign_id: str | None = None,
) -> None:
    verify_websocket_api_key(websocket)
    await websocket.accept()

    broadcaster = getattr(websocket.app.state, "event_broadcaster", None)
    if broadcaster is None:
        await websocket.send_json({"event_type": "stream.unavailable", "payload": {}})
        await websocket.close()
        return

    queue = broadcaster.subscribe(campaign_id)
    await websocket.send_json(
        {
            "event_type": "stream.connected",
            "payload": {"campaign_id": campaign_id},
        }
    )
    try:
        while True:
            event = await queue.get()
            await websocket.send_json(
                {
                    "event_type": event.event_type,
                    "payload": event.payload,
                    "created_at": event.created_at.isoformat(),
                }
            )
    except WebSocketDisconnect:
        pass
    finally:
        broadcaster.unsubscribe(queue, campaign_id)
