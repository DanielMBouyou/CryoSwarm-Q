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
    """Bridge the blockade interaction graph to a MIS-style QUBO encoding."""

    @property
    def available(self) -> bool:
        return QOOLQIT_AVAILABLE

    def status(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "warning": QOOLQIT_IMPORT_ERROR,
        }

    def build_qubo_from_graph(
        self,
        interaction_graph: list[list[bool]],
        weights: list[float] | None = None,
    ) -> dict[str, Any]:
        """Encode a blockade graph as a QUBO for maximum independent set."""
        if not self.available:
            return {
                "status": "unavailable",
                "warning": QOOLQIT_IMPORT_ERROR,
            }

        n_vertices = len(interaction_graph)
        penalty = 2.0
        qubo: dict[tuple[int, int], float] = {}

        for vertex in range(n_vertices):
            weight = weights[vertex] if weights else 1.0
            qubo[(vertex, vertex)] = -float(weight)

        for row_index in range(n_vertices):
            for col_index in range(row_index + 1, n_vertices):
                if interaction_graph[row_index][col_index]:
                    qubo[(row_index, col_index)] = penalty

        return {
            "status": "available",
            "qubo_matrix": qubo,
            "n_variables": n_vertices,
            "penalty": penalty,
        }

    def build_problem(self, specification: dict[str, Any]) -> dict[str, Any]:
        if not self.available:
            return {
                "available": False,
                "message": "QoolQit is not installed. Using structured placeholder only.",
                "specification": specification,
            }
        return {
            "available": True,
            "message": "QoolQit adapter is available for QUBO-style compilation.",
            "specification": specification,
        }
