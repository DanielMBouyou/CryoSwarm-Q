"""Pre-formatted LaTeX formula collections for the CryoSwarm-Q dashboard.

Each function returns a ``dict[str, str]`` mapping formula names to raw
LaTeX strings.  Pages choose how to render them (``st.latex`` or inline
``$...$`` via ``st.markdown``).
"""
from __future__ import annotations


def hamiltonian_formulas() -> dict[str, str]:
    """Rydberg Hamiltonian and interaction terms."""
    return {
        "full_hamiltonian": (
            r"\hat{H} = \frac{\Omega}{2}\sum_{i=1}^{N} \hat{\sigma}_x^{(i)}"
            r" - \delta\sum_{i=1}^{N} \hat{n}_i"
            r" + \sum_{i<j} \frac{C_6}{|\mathbf{r}_i - \mathbf{r}_j|^6}\,\hat{n}_i\hat{n}_j"
        ),
        "interaction": r"U_{ij} = \frac{C_6}{r_{ij}^6}",
        "blockade_radius": r"R_b = \left(\frac{C_6}{\Omega}\right)^{\!1/6}",
        "c6_value": (
            r"C_6 = 2\pi \times 862\,690 \;\text{rad}\cdot\mu\text{m}^6/\mu\text{s}"
            r" \quad (^{87}\text{Rb},\; |70S_{1/2}\rangle)"
        ),
        "spectral_gap": r"\Delta E = E_1 - E_0",
        "ipr": r"\text{IPR} = \sum_{n} |\langle n | \psi_0 \rangle|^4",
        "adiabatic_condition": (
            r"\frac{|\langle \psi_1 | \dot{H} | \psi_0 \rangle|}{(\Delta E)^2} \ll 1"
        ),
    }


def observable_formulas() -> dict[str, str]:
    """Quantum observables computed from state vectors."""
    return {
        "rydberg_density": r"\langle \hat{n}_i \rangle = \text{Tr}\!\left(\hat{n}_i |\psi\rangle\langle\psi|\right)",
        "connected_correlation": (
            r"g_{ij}^{(2)} = \langle \hat{n}_i \hat{n}_j \rangle"
            r" - \langle \hat{n}_i \rangle \langle \hat{n}_j \rangle"
        ),
        "antiferromagnetic_order": (
            r"m_{\text{AF}} = \frac{1}{N}\left|\sum_{i=1}^{N} (-1)^i"
            r" \bigl(2\langle \hat{n}_i \rangle - 1\bigr)\right|"
        ),
        "entanglement_entropy": r"S_A = -\text{Tr}\!\left(\rho_A \log_2 \rho_A\right)",
        "total_rydberg_fraction": r"\bar{n} = \frac{1}{N}\sum_{i=1}^{N} \langle \hat{n}_i \rangle",
        "state_fidelity": r"\mathcal{F} = |\langle \psi_1 | \psi_2 \rangle|^2",
        "bitstring_probability": (
            r"P(b_0 b_1 \cdots b_{N-1}) = |\langle b_0 b_1 \cdots b_{N-1} | \psi \rangle|^2"
        ),
    }


def mis_formulas() -> dict[str, str]:
    """Maximum Independent Set on the blockade graph."""
    return {
        "mis_definition": (
            r"\text{MIS}(G) = \arg\max_{S \subseteq V}\; |S|"
            r" \quad \text{s.t.} \quad \forall\,(u,v)\in E,\;"
            r" u \notin S \;\text{or}\; v \notin S"
        ),
        "mis_overlap": r"P_{\text{MIS}} = \sum_{s\,\in\,\text{MIS}} |\langle s | \psi_0 \rangle|^2",
        "mis_cost_function": (
            r"C_{\text{MIS}} = -\sum_{i \in V} n_i"
            r" + \alpha \sum_{(i,j) \in E} n_i n_j, \quad \alpha > 1"
        ),
    }


def pulse_formulas() -> dict[str, str]:
    """Pulse sequence waveforms and time evolution."""
    return {
        "blackman": (
            r"\Omega_{\text{Blackman}}(t) = \Omega_{\max}\!\left[0.42"
            r" - 0.50\cos\!\left(\frac{2\pi t}{T}\right)"
            r" + 0.08\cos\!\left(\frac{4\pi t}{T}\right)\right]"
        ),
        "linear_sweep": (
            r"\delta(t) = \delta_{\text{start}}"
            r" + \frac{\delta_{\text{end}} - \delta_{\text{start}}}{T}\,t"
        ),
        "pulse_area": r"\theta = \int_0^T \Omega(t)\,dt \qquad (\pi\text{-pulse:}\;\theta = \pi)",
        "time_evolution": (
            r"|\psi(t)\rangle = \mathcal{T}\exp\!\left(-i\int_0^t"
            r" \hat{H}(t')\,dt'\right)|\psi(0)\rangle"
        ),
        "trotter_suzuki_2": (
            r"e^{-i(A+B)\Delta t} = e^{-iA\,\Delta t/2}\,e^{-iB\,\Delta t}"
            r"\,e^{-iA\,\Delta t/2} + \mathcal{O}(\Delta t^3)"
        ),
        "ramp": r"\Omega_{\text{ramp}}(t) = \Omega_{\max} \cdot \frac{t}{T}",
        "constant": r"\Omega_{\text{const}}(t) = \Omega_{\max}, \quad \delta(t) = \delta_0",
    }


def robustness_formulas() -> dict[str, str]:
    """Robustness scoring and noise modeling."""
    return {
        "robustness_score": (
            r"S_{\text{robust}} = w_n \, s_{\text{nom}}"
            r" + w_a \, \bar{s}_{\text{pert}}"
            r" + w_w \, s_{\text{worst}}"
            r" + w_s \, b_{\text{stab}}"
        ),
        "robustness_weights": r"w_n = 0.25,\quad w_a = 0.35,\quad w_w = 0.30,\quad w_s = 0.10",
        "stability_bonus": (
            r"b_{\text{stab}} = \max\!\left(0,\;"
            r" 1 - \frac{\sigma_s}{\sigma_{\text{thresh}}}\right)"
        ),
        "penalty": (
            r"\text{Penalty} = \max\!\left(0,\;"
            r" (s_{\text{nom}} - s_{\text{worst}}) + \sigma_s\right)"
        ),
        "objective_score": (
            r"S_{\text{obj}} = \alpha\, s_{\text{obs}}"
            r" + \beta\, s_{\text{robust}}"
            r" - \gamma\, c_{\text{exec}}"
            r" - \delta\, \ell_{\text{latency}}"
        ),
        "weight_constraint": (
            r"\alpha + \beta + \gamma + \delta = 1"
            r" \quad (\alpha{=}0.45,\;\beta{=}0.35,\;\gamma{=}0.10,\;\delta{=}0.10)"
        ),
        "noise_amplitude": (
            r"\Omega \to \Omega\,(1 + \epsilon_\Omega),"
            r" \quad \epsilon_\Omega \sim \mathcal{N}(0, \sigma_{\text{amp}}^2)"
        ),
        "noise_detuning": (
            r"\delta \to \delta + \epsilon_\delta,"
            r" \quad \epsilon_\delta \sim \mathcal{N}(0, \sigma_{\text{det}}^2 |\delta| + 0.1)"
        ),
        "dephasing_lindblad": (
            r"\mathcal{L}_\phi[\rho] = \gamma_\phi"
            r" \left(\hat{n}\,\rho\,\hat{n}"
            r" - \tfrac{1}{2}\{\hat{n}^2, \rho\}\right)"
        ),
        "density_score": (
            r"s_{\text{dens}} = \max\!\left(0,\;"
            r" 1 - \frac{|\bar{n}_{\text{obs}} - \bar{n}_{\text{target}}|}"
            r"{\max(\bar{n}_{\text{target}},\, 1 - \bar{n}_{\text{target}},\, 0.5)}\right)"
        ),
        "blockade_score": (
            r"s_{\text{block}} = \max\!\left(0,\;"
            r" 1 - \min\!\left(\frac{v_{\text{block}}}{0.20},\, 1\right)\right)"
        ),
    }


def ml_formulas() -> dict[str, str]:
    """Machine learning: PPO, surrogate, bandit."""
    return {
        "ucb1": r"\text{UCB1}(s) = \bar{r}_s + \sqrt{\frac{2\ln N}{n_s}}",
        "ppo_objective": (
            r"L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}\!\left["
            r"\min\!\left(r_t(\theta)\,\hat{A}_t,\;"
            r"\text{clip}(r_t(\theta),\, 1 {\pm} \epsilon)\,\hat{A}_t\right)\right]"
        ),
        "importance_ratio": (
            r"r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}"
            r"{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}"
        ),
        "gae": (
            r"\hat{A}_t^{\text{GAE}} = \sum_{\ell=0}^{\infty}"
            r"(\gamma\lambda)^\ell\,\delta_{t+\ell},"
            r" \quad \delta_t = r_t + \gamma\,V(s_{t+1}) - V(s_t)"
        ),
        "ensemble_mean": r"\hat{y}_{\text{ens}} = \frac{1}{M}\sum_{m=1}^{M} f_m(\mathbf{x})",
        "epistemic_uncertainty": (
            r"\sigma_{\text{epist}}^2 = \frac{1}{M}\sum_{m=1}^{M}"
            r" \bigl(f_m(\mathbf{x}) - \hat{y}_{\text{ens}}\bigr)^2"
        ),
        "weighted_mse": (
            r"L = \frac{1}{B}\sum_{b=1}^{B}\sum_{k=1}^{K}"
            r" w_k\,\bigl(\hat{y}_{b,k} - y_{b,k}\bigr)^2"
        ),
    }


def campaign_formulas() -> dict[str, str]:
    """Campaign analytics: Pareto, regret, ranking."""
    return {
        "pareto_optimal": (
            r"\mathbf{x}^* \text{ is Pareto optimal if }"
            r" \nexists\,\mathbf{x} : f_i(\mathbf{x}) \geq f_i(\mathbf{x}^*)"
            r"\;\forall i,\; f_j(\mathbf{x}) > f_j(\mathbf{x}^*)\;\exists j"
        ),
        "cumulative_regret": r"R_T = \sum_{t=1}^{T}\!\left(r^* - r_t\right)",
        "acquisition_ucb": r"\alpha(\mathbf{x}) = \mu(\mathbf{x}) + \kappa\,\sigma(\mathbf{x})",
        "ranking_key": (
            r"\text{rank}(c) = \text{sort}\!\left("
            r"-S_{\text{obj}},\, -s_{\text{worst}},\, -S_{\text{robust}},\, -s_{\text{nom}}\right)"
        ),
    }
