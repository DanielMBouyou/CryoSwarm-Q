from __future__ import annotations

from typing import Any


try:
    from pulser import Register

    PULSER_AVAILABLE = True
    PULSER_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - depends on external package
    Register = None
    PULSER_AVAILABLE = False
    PULSER_IMPORT_ERROR = str(exc)


def create_simple_register(coordinates: list[tuple[float, float]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "pulser_available": PULSER_AVAILABLE,
        "coordinate_count": len(coordinates),
        "coordinates": coordinates,
    }

    if not PULSER_AVAILABLE:
        summary["warning"] = PULSER_IMPORT_ERROR or "Pulser is not installed."
        return summary

    try:  # pragma: no cover - requires pulser runtime
        if hasattr(Register, "from_coordinates"):
            register = Register.from_coordinates(coordinates, prefix="q")
        else:
            register = Register({f"q{i}": point for i, point in enumerate(coordinates)})
        summary["pulser_register_created"] = True
        summary["register_repr"] = str(register)
    except Exception as exc:
        summary["pulser_register_created"] = False
        summary["warning"] = f"Pulser register creation failed: {exc}"
    return summary


def build_simple_sequence_summary(
    coordinates: list[tuple[float, float]],
    duration_ns: int,
    amplitude: float,
    detuning: float,
    phase: float,
) -> dict[str, Any]:
    return {
        "pulser_available": PULSER_AVAILABLE,
        "register_summary": create_simple_register(coordinates),
        "duration_ns": duration_ns,
        "controls": {
            "Omega(t)": amplitude,
            "delta(t)": detuning,
            "phi(t)": phase,
        },
        "serialization_mode": "structured_summary",
    }


def summarize_sequence_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {"available": False}
    return {
        "pulser_available": payload.get("pulser_available", False),
        "duration_ns": payload.get("duration_ns"),
        "controls": payload.get("controls", {}),
        "serialization_mode": payload.get("serialization_mode", "structured_summary"),
    }
