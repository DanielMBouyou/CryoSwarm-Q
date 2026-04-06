"""CryoSwarm-Q machine learning modules.

Three phases:
- Phase 1: Surrogate model (MLP) that predicts robustness scores from
  experiment configurations, used as a fast pre-filter in the pipeline.
- Phase 2: Reinforcement learning agent (PPO) that learns to generate
  pulse-sequence parameters directly.
- Phase 3: GPU-accelerated simulation backend and distributed training.

All phases are opt-in.  Import guards ensure the system degrades
gracefully when PyTorch is not installed.
"""
