from packages.pasqal_adapters.pulser_adapter import summarize_register_physics


def test_register_physics_summary_detects_valid_spacing() -> None:
    summary = summarize_register_physics([(0.0, 0.0), (0.0, 7.0), (7.0, 0.0)])

    assert summary.valid is True
    assert summary.min_distance_um >= 5.0
    assert summary.blockade_radius_um > 0.0
    assert summary.blockade_pair_count >= 1
