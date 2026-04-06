from __future__ import annotations

import numpy as np

from packages.core.enums import SequenceFamily
from packages.core.models import RegisterCandidate, SequenceCandidate
from packages.core.parameter_space import PhysicsParameterSpace
from packages.ml.dataset import INPUT_DIM_V2, build_feature_vector_v2


def test_feature_vector_v2_shape_and_bounds() -> None:
    param_space = PhysicsParameterSpace.default()
    register = RegisterCandidate(
        campaign_id="camp_test",
        spec_id="spec_test",
        label="line-4-s7.0",
        layout_type="line",
        atom_count=4,
        coordinates=[(0.0, 0.0), (7.0, 0.0), (14.0, 0.0), (21.0, 0.0)],
        min_distance_um=7.0,
        blockade_radius_um=8.5,
        blockade_pair_count=3,
        van_der_waals_matrix=[[0.0] * 4 for _ in range(4)],
        feasibility_score=0.82,
        reasoning_summary="register",
        metadata={"spacing_um": 7.0},
    )
    sequence = SequenceCandidate(
        campaign_id="camp_test",
        spec_id="spec_test",
        register_candidate_id=register.id,
        label="adiabatic-seq",
        sequence_family=SequenceFamily.ADIABATIC_SWEEP,
        duration_ns=3000,
        amplitude=5.0,
        detuning=-20.0,
        phase=0.0,
        waveform_kind="adiabatic_conservative",
        predicted_cost=0.2,
        reasoning_summary="sequence",
        metadata={"detuning_end": 10.0},
    )

    features = build_feature_vector_v2(register, sequence, param_space)

    assert features.shape == (INPUT_DIM_V2,)
    assert features.dtype == np.float32
    assert np.all(features >= 0.0)
    assert features[8] >= 0.0  # omega_over_interaction_norm
    assert features[9] >= 0.0  # detuning_over_omega_norm
    assert features[-1] > 0.0  # sweep_span_norm
