from __future__ import annotations

import packages.pasqal_adapters.qoolqit_adapter as qoolqit_module
from packages.pasqal_adapters.qoolqit_adapter import QoolQitAdapter


def test_qubo_construction_from_interaction_graph(monkeypatch) -> None:
    monkeypatch.setattr(qoolqit_module, "QOOLQIT_AVAILABLE", True)
    adapter = QoolQitAdapter()
    graph = [
        [False, True, False],
        [True, False, True],
        [False, True, False],
    ]

    result = adapter.build_qubo_from_graph(graph)

    assert result["status"] == "available"
    assert result["n_variables"] == 3
    assert result["qubo_matrix"][(0, 1)] > 0
    assert result["qubo_matrix"][(1, 2)] > 0


def test_qubo_diagonal_terms_are_negative(monkeypatch) -> None:
    monkeypatch.setattr(qoolqit_module, "QOOLQIT_AVAILABLE", True)
    adapter = QoolQitAdapter()
    graph = [
        [False, False],
        [False, False],
    ]

    result = adapter.build_qubo_from_graph(graph, weights=[1.0, 1.5])

    assert result["qubo_matrix"][(0, 0)] < 0
    assert result["qubo_matrix"][(1, 1)] < 0


def test_connected_pairs_receive_positive_penalty(monkeypatch) -> None:
    monkeypatch.setattr(qoolqit_module, "QOOLQIT_AVAILABLE", True)
    adapter = QoolQitAdapter()
    graph = [
        [False, True, True],
        [True, False, False],
        [True, False, False],
    ]

    result = adapter.build_qubo_from_graph(graph)

    assert result["qubo_matrix"][(0, 1)] == result["penalty"]
    assert result["qubo_matrix"][(0, 2)] == result["penalty"]
