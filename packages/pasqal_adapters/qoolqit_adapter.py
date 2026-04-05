from __future__ import annotations

from typing import Any


try:
    import qoolqit  # type: ignore

    QOOLQIT_AVAILABLE = True
    QOOLQIT_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - optional dependency
    qoolqit = None
    QOOLQIT_AVAILABLE = False
    QOOLQIT_IMPORT_ERROR = str(exc)


class QoolQitAdapter:
    def status(self) -> dict[str, Any]:
        return {
            "available": QOOLQIT_AVAILABLE,
            "warning": QOOLQIT_IMPORT_ERROR,
        }

    def build_problem(self, specification: dict[str, Any]) -> dict[str, Any]:
        if not QOOLQIT_AVAILABLE:
            return {
                "available": False,
                "message": "QoolQit is not installed. Using structured placeholder only.",
                "specification": specification,
            }
        return {
            "available": True,
            "message": "QoolQit adapter is available for future problem compilation.",
            "specification": specification,
        }
