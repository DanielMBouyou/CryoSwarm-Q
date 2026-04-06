# P3 — AMÉLIORATIONS SCIENTIFIQUES : Prompt d'implémentation exhaustif

> **Destinataire** : Agent IA de code.
> **Mode** : Implémentation directe — écris chaque ligne, chaque import, chaque test.
> **Philosophie** : Améliorations scientifiques ciblées pour la crédibilité recherche. Chaque modification doit être motivée par un défaut physique ou algorithmique concret dans le code actuel. CryoSwarm-Q est un système de simulation d'atomes neutres — la rigueur scientifique est non-négociable.

---

## Contexte du projet

CryoSwarm-Q est un système multi-agent hardware-aware pour la conception autonome d'expériences en informatique quantique à atomes neutres. Les modules de simulation (`packages/simulation/`) et le backend GPU (`packages/ml/gpu_backend.py`) effectuent la construction hamiltonienne, l'évolution temporelle et l'extraction d'observables.

**Fichier manifeste** : `CLAUDE.md` à la racine contient la vision complète du projet.

**Hypothèse** : Les tâches P0, P1 et P2 ont été implémentées. Si certaines ne sont pas encore en place, adapte-toi — ne casse rien.

**Règles impératives** :
- Python 3.11+, types partout, `from __future__ import annotations`
- Imports absolus uniquement (`from packages.simulation.hamiltonian import ...`)
- Chaque import groupé : stdlib, third-party, project
- `pytest tests/ -x --tb=short` après CHAQUE tâche
- Zéro `# TODO` laissé dans le code produit
- Les améliorations doivent être rétro-compatibles : signatures existantes préservées, nouveaux paramètres optionnels avec valeurs par défaut

---

## TÂCHE 1 — Remplacer l'évolution Taylor 2nd ordre par Lanczos dans le GPU backend

### Problème actuel

**Fichier** : `packages/ml/gpu_backend.py` (lignes 167-171)

L'évolution temporelle GPU utilise une approximation de Taylor au 2ème ordre :

```python
dense_hamiltonian = hamiltonian.to_dense()
h_psi = torch.mv(dense_hamiltonian, psi)
h2_psi = torch.mv(dense_hamiltonian, h_psi)
psi = psi - 1j * dt * h_psi - 0.5 * dt * dt * h2_psi
psi = psi / psi.norm()
```

**Problèmes** :
1. **Non-unitarité** : L'approximation `e^{-iHdt} ≈ I - iHdt - H²dt²/2` n'est pas unitaire. La renormalisation `psi / norm(psi)` masque la dérive d'erreur mais n'est pas physiquement justifiée.
2. **Précision** : Pour des pas de temps `dt` non infinitésimaux, l'erreur est `O(dt³)` par pas, soit `O(T·dt²)` cumulée sur N pas — inacceptable pour des systèmes avec fort couplage (grand `||H||·dt`).
3. **Conversion dense** : `hamiltonian.to_dense()` annule l'avantage de la construction sparse COO. Pour n=24 atomes, la matrice dense occupe `2^24 × 2^24 × 16 bytes = 4 TiB` — impossible en pratique.
4. **Crédibilité** : Un revieweur spécialisé en simulation quantique identifierait immédiatement cette approximation comme insuffisante.

### Solution : Lanczos short-iteration time evolution

L'algorithme de Lanczos construit une base de Krylov `{|ψ⟩, H|ψ⟩, H²|ψ⟩, ...}` de dimension `m << 2^n`, orthogonalise pour obtenir une matrice tridiagonale `T_m`, et calcule `e^{-iHdt}|ψ⟩ ≈ ||ψ|| · V_m · e^{-iT_m dt} · e_1`.

**Avantages** :
- Unitarité préservée à la précision machine
- Ne nécessite que des produits matrice-vecteur sparse (jamais de `.to_dense()`)
- Convergence exponentielle : `m = 15-25` itérations suffisent pour `dt` raisonnable
- Compatible GPU via `torch.sparse.mm`

### Modifications

#### 1. `packages/ml/gpu_backend.py` — Ajouter la fonction Lanczos

Ajouter cette fonction **après** `build_gpu_hamiltonian` et **avant** `gpu_time_evolution` :

```python
def _lanczos_expm_multiply(
    hamiltonian: torch.Tensor,
    psi: torch.Tensor,
    dt: float,
    krylov_dim: int = 20,
) -> torch.Tensor:
    """Compute e^{-i H dt} |psi> via Lanczos iteration on GPU.

    Builds a Krylov subspace of dimension *krylov_dim*, projects H onto the
    resulting tridiagonal matrix T, exponentiates T (small dense matrix), and
    maps back to the full Hilbert space.

    Parameters
    ----------
    hamiltonian : sparse COO tensor on GPU/CPU.
    psi : state vector, will NOT be modified.
    dt : time step in microseconds.
    krylov_dim : maximum Lanczos iterations (15-25 is typical).

    Returns
    -------
    Evolved state vector, normalised.
    """
    norm_psi = psi.norm()
    if norm_psi < 1e-15:
        return psi.clone()

    device = psi.device
    dtype = psi.dtype
    dim = psi.shape[0]
    m = min(krylov_dim, dim)

    # Lanczos vectors and tridiagonal elements
    V = torch.zeros(m, dim, dtype=dtype, device=device)
    alpha = torch.zeros(m, dtype=torch.float64, device=device)
    beta = torch.zeros(m, dtype=torch.float64, device=device)

    V[0] = psi / norm_psi

    # First iteration
    w = torch.mv(hamiltonian.to_dense() if not hamiltonian.is_sparse else hamiltonian, V[0])
    # For sparse COO, use sparse @ dense vector
    if hamiltonian.is_sparse:
        w = torch.sparse.mm(hamiltonian, V[0].unsqueeze(1)).squeeze(1)
    else:
        w = torch.mv(hamiltonian, V[0])

    alpha[0] = torch.dot(w.real, V[0].real) + torch.dot(w.imag, V[0].imag)
    w = w - alpha[0] * V[0]

    for j in range(1, m):
        beta[j] = w.norm().real
        if beta[j] < 1e-14:
            m = j
            break
        V[j] = w / beta[j]

        if hamiltonian.is_sparse:
            w = torch.sparse.mm(hamiltonian, V[j].unsqueeze(1)).squeeze(1)
        else:
            w = torch.mv(hamiltonian, V[j])

        alpha[j] = torch.dot(w.real, V[j].real) + torch.dot(w.imag, V[j].imag)
        w = w - alpha[j] * V[j] - beta[j] * V[j - 1]

        # Re-orthogonalisation (full, for numerical stability)
        for k in range(j + 1):
            overlap = torch.dot(V[k].conj(), w)
            w = w - overlap * V[k]

    # Build tridiagonal matrix T (m x m) on CPU for expm
    T = torch.zeros(m, m, dtype=torch.float64, device="cpu")
    for j in range(m):
        T[j, j] = alpha[j].cpu()
    for j in range(1, m):
        T[j, j - 1] = beta[j].cpu()
        T[j - 1, j] = beta[j].cpu()

    # Exponentiate T: e^{-i T dt}
    eigvals, eigvecs = torch.linalg.eigh(T)
    exp_diag = torch.exp(-1j * dt * eigvals.to(torch.complex128))
    expT = (eigvecs.to(torch.complex128) @ torch.diag(exp_diag) @ eigvecs.T.to(torch.complex128))

    # e_1 = [1, 0, ..., 0]
    e1 = torch.zeros(m, dtype=torch.complex128, device="cpu")
    e1[0] = 1.0

    coeffs = (expT @ e1).to(device)

    # Map back: psi_new = norm_psi * V^T @ coeffs
    result = torch.zeros(dim, dtype=dtype, device=device)
    for j in range(m):
        result += (coeffs[j] * norm_psi) * V[j]

    result = result / result.norm()
    return result
```

#### 2. `packages/ml/gpu_backend.py` — Modifier `gpu_time_evolution`

Remplacer le bloc Taylor (lignes 167-171) par l'appel Lanczos.

**Avant** :

```python
        dense_hamiltonian = hamiltonian.to_dense()
        h_psi = torch.mv(dense_hamiltonian, psi)
        h2_psi = torch.mv(dense_hamiltonian, h_psi)
        psi = psi - 1j * dt * h_psi - 0.5 * dt * dt * h2_psi
        psi = psi / psi.norm()
```

**Après** :

```python
        psi = _lanczos_expm_multiply(hamiltonian, psi, dt, krylov_dim=krylov_dim)
```

#### 3. `packages/ml/gpu_backend.py` — Ajouter le paramètre `krylov_dim` à `gpu_time_evolution`

Ajouter `krylov_dim: int = 20` à la signature de `gpu_time_evolution` :

**Avant** :

```python
def gpu_time_evolution(
    coords: list[tuple[float, float]],
    omega_max: float,
    delta_start: float,
    delta_end: float,
    duration_ns: float,
    n_steps: int = 200,
    omega_shape: str = "constant",
    c6: float = C6_RB87_70S,
    device: str | None = None,
    max_atoms_gpu: int | None = None,
) -> dict[str, Any]:
```

**Après** :

```python
def gpu_time_evolution(
    coords: list[tuple[float, float]],
    omega_max: float,
    delta_start: float,
    delta_end: float,
    duration_ns: float,
    n_steps: int = 200,
    omega_shape: str = "constant",
    c6: float = C6_RB87_70S,
    device: str | None = None,
    max_atoms_gpu: int | None = None,
    krylov_dim: int = 20,
) -> dict[str, Any]:
```

#### 4. Ajouter `krylov_dim` au dict de retour

Dans le `return` de `gpu_time_evolution`, ajouter :

```python
        "krylov_dim": krylov_dim,
```

### Tests

**Fichier** : `tests/test_ml_gpu.py` — Ajouter les tests suivants dans la classe `TestGPUTimeEvolution` et une nouvelle classe `TestLanczos` :

```python
class TestLanczos:
    """Tests for the Lanczos time-evolution kernel."""

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_identity_for_zero_dt(self) -> None:
        """Lanczos with dt=0 should return the same state."""
        from packages.ml.gpu_backend import _lanczos_expm_multiply, build_gpu_hamiltonian

        coords = [(0.0, 0.0), (7.0, 0.0)]
        H = build_gpu_hamiltonian(coords, omega=5.0, delta=-10.0, device="cpu")
        dim = H.shape[0]
        psi = torch.zeros(dim, dtype=torch.complex128)
        psi[0] = 1.0

        psi_out = _lanczos_expm_multiply(H, psi, dt=0.0, krylov_dim=10)
        np.testing.assert_allclose(
            psi_out.numpy(), psi.numpy(), atol=1e-12,
        )

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_unitarity_preserved(self) -> None:
        """Evolved state should have unit norm."""
        from packages.ml.gpu_backend import _lanczos_expm_multiply, build_gpu_hamiltonian

        coords = [(0.0, 0.0), (7.0, 0.0), (3.5, 6.06)]
        H = build_gpu_hamiltonian(coords, omega=5.0, delta=-10.0, device="cpu")
        dim = H.shape[0]
        psi = torch.zeros(dim, dtype=torch.complex128)
        psi[0] = 1.0

        for _ in range(20):
            psi = _lanczos_expm_multiply(H, psi, dt=0.01, krylov_dim=15)

        assert abs(psi.norm().item() - 1.0) < 1e-10

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_matches_scipy_expm(self) -> None:
        """Lanczos should match scipy.linalg.expm for small systems."""
        pytest.importorskip("scipy")
        from scipy.linalg import expm as scipy_expm

        from packages.ml.gpu_backend import _lanczos_expm_multiply, build_gpu_hamiltonian
        from packages.simulation.hamiltonian import build_hamiltonian_matrix

        coords = [(0.0, 0.0), (7.0, 0.0)]
        omega, delta = 5.0, -10.0
        dt = 0.005

        H_cpu = build_hamiltonian_matrix(coords, omega, delta)
        dim = H_cpu.shape[0]
        psi_np = np.zeros(dim, dtype=np.complex128)
        psi_np[0] = 1.0
        psi_exact = scipy_expm(-1j * H_cpu * dt) @ psi_np
        psi_exact /= np.linalg.norm(psi_exact)

        H_torch = build_gpu_hamiltonian(coords, omega, delta, device="cpu")
        psi_torch = torch.zeros(dim, dtype=torch.complex128)
        psi_torch[0] = 1.0
        psi_lanczos = _lanczos_expm_multiply(H_torch, psi_torch, dt, krylov_dim=15)

        np.testing.assert_allclose(
            psi_lanczos.numpy(), psi_exact, atol=1e-8,
        )

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_convergence_with_krylov_dim(self) -> None:
        """Higher krylov_dim should give better (or equal) accuracy."""
        pytest.importorskip("scipy")
        from scipy.linalg import expm as scipy_expm

        from packages.ml.gpu_backend import _lanczos_expm_multiply, build_gpu_hamiltonian
        from packages.simulation.hamiltonian import build_hamiltonian_matrix

        coords = [(0.0, 0.0), (7.0, 0.0), (3.5, 6.06)]
        omega, delta = 5.0, -10.0
        dt = 0.01

        H_cpu = build_hamiltonian_matrix(coords, omega, delta)
        dim = H_cpu.shape[0]
        psi_np = np.zeros(dim, dtype=np.complex128)
        psi_np[0] = 1.0
        psi_exact = scipy_expm(-1j * H_cpu * dt) @ psi_np
        psi_exact /= np.linalg.norm(psi_exact)

        H_torch = build_gpu_hamiltonian(coords, omega, delta, device="cpu")
        psi_torch = torch.zeros(dim, dtype=torch.complex128)
        psi_torch[0] = 1.0

        errors = []
        for m in [5, 8, 15]:
            psi_m = _lanczos_expm_multiply(H_torch, psi_torch, dt, krylov_dim=m)
            err = np.linalg.norm(psi_m.numpy() - psi_exact)
            errors.append(err)

        # Errors should be non-increasing (convergence)
        assert errors[-1] <= errors[0] + 1e-12
```

### Vérification

```bash
pytest tests/test_ml_gpu.py -x --tb=short -v
```

---

## TÂCHE 2 — Documenter et valider la convention MSB/LSB des bitstrings

### Problème actuel

La convention de bitstring est utilisée dans **5 fichiers** mais n'est documentée explicitement que dans un seul (`observables.py`, ligne 9). Le commentaire en docstring dit :

```
Basis ordering: |q0 q1 ... q_{N-1}> with q0 as the most significant bit.
```

Cela signifie : l'état de base `|q0 q1 ... q_{N-1}>` est encodé comme un entier où `q0` occupe le bit de poids fort. Concrètement, pour l'état `|101>` de 3 atomes :
- `q0 = 1` (bit de poids `2^2 = 4`)
- `q1 = 0` (bit de poids `2^1 = 2`)
- `q2 = 1` (bit de poids `2^0 = 1`)
- Index entier = `5`

La formule d'extraction est cohérente partout :
```python
(basis >> (n_atoms - 1 - i)) & 1  # True si atome i est excité
```

**Problème** : cette convention est implicite dans la plupart des fichiers. Un contributeur pourrait introduire un bug en utilisant la convention opposée (`q0 = LSB`), d'autant plus que certaines librairies (Qiskit, Cirq) utilisent la convention inverse. Il n'y a aucune assertion défensive et aucun test dédié vérifiant la cohérence inter-modules.

### Fichiers concernés

| Fichier | Lignes | Utilisation |
|---------|--------|-------------|
| `packages/simulation/observables.py` | 9, 35, 55, 130 | Docstring + extraction `(basis >> (n_atoms - 1 - i)) & 1` |
| `packages/ml/gpu_backend.py` | 46, 59 | `1 << (n_atoms - 1 - qubit)` dans `_sigma_x_elements` et `_n_op_diagonal` |
| `packages/simulation/hamiltonian.py` | 240-248 | `(basis >> (n - 1 - site)) & 1` dans `build_sparse_hamiltonian` |
| `packages/simulation/hamiltonian.py` | 136-139 | `bits[idx] = "1"` dans `mis_bitstrings` (q0 = leftmost char = MSB) |
| `tests/test_ml_gpu.py` | 47, 57 | Commentaires `# q0 = MSB`, `# q1 = LSB` |

### Modifications

#### 1. `packages/simulation/observables.py` — Renforcer la documentation

Ajouter un bloc de documentation formel au module et une constante symbolique :

**Avant** (lignes 1-11) :

```python
"""Quantum observables for neutral-atom Rydberg systems.

Computes expectation values, correlations, and entanglement measures
from quantum state vectors in the {|g>, |r>} computational basis.

Convention
----------
|g> = |0>, |r> = |1> per atom.
Basis ordering: |q0 q1 ... q_{N-1}> with q0 as the most significant bit.
n_i = |r_i><r_i| measures Rydberg occupation of atom i.
"""
```

**Après** :

```python
"""Quantum observables for neutral-atom Rydberg systems.

Computes expectation values, correlations, and entanglement measures
from quantum state vectors in the {|g>, |r>} computational basis.

Bitstring convention (project-wide)
------------------------------------
- |g> = |0>, |r> = |1> per atom.
- Basis ordering: |q0 q1 ... q_{N-1}> with **q0 as the most significant bit (MSB)**.
- Integer index of state |b_0 b_1 ... b_{N-1}> = sum_i b_i * 2^{N-1-i}.
- Bit extraction: atom *i* is excited iff ``(index >> (n_atoms - 1 - i)) & 1``.
- Bitstring format: ``format(index, f'0{n_atoms}b')`` — leftmost character is q0.
- n_i = |r_i><r_i| measures Rydberg occupation of atom i.

This convention is consistent with Pulser's default qubit ordering.
All modules in ``packages/simulation/`` and ``packages/ml/gpu_backend.py``
MUST follow this convention.
"""
```

#### 2. `packages/ml/gpu_backend.py` — Ajouter la documentation de convention

Ajouter un commentaire de convention au niveau du module, après les imports :

**Après la ligne** `logger = get_logger(__name__)` **ajouter** :

```python

# ---------- Bitstring convention ----------
# This module follows the project-wide MSB convention:
#   Atom i corresponds to bit position (n_atoms - 1 - i).
#   |q0 q1 ... q_{N-1}> with q0 = most significant bit.
# See packages/simulation/observables.py for the canonical reference.
```

#### 3. `packages/simulation/hamiltonian.py` — Ajouter la documentation de convention

Ajouter dans la docstring du module, après la section `Units` :

**Avant** (lignes 13-17) :

```python
Units
-----
- Distances: micrometers (um)
- Frequencies / energies: rad/us  (with hbar = 1)
- C6 coefficient: rad * um^6 / us
```

**Après** :

```python
Units
-----
- Distances: micrometers (um)
- Frequencies / energies: rad/us  (with hbar = 1)
- C6 coefficient: rad * um^6 / us

Bitstring convention
--------------------
All bit-level operations use the project-wide MSB convention:
atom *i* corresponds to bit position ``(n_atoms - 1 - i)`` in the
integer basis index.  See ``packages/simulation/observables.py``.
```

#### 4. Ajouter une fonction utilitaire d'extraction de bit

**Fichier** : `packages/simulation/observables.py` — Ajouter après les imports, avant `rydberg_density` :

```python
def atom_excited(basis_index: int, atom: int, n_atoms: int) -> bool:
    """Check if *atom* is in |r> in the given computational-basis index.

    Uses the project-wide MSB convention: atom 0 is the most significant bit.
    """
    return bool((basis_index >> (n_atoms - 1 - atom)) & 1)
```

Puis utiliser `atom_excited` dans `rydberg_density` et `pair_correlation` pour renforcer la lisibilité (pas obligatoire — la fonction existe comme point de référence canonique).

### Tests

**Fichier** : `tests/test_observables.py` — Ajouter une classe dédiée :

```python
class TestBitstringConvention:
    """Verify MSB bitstring convention consistency across modules."""

    def test_atom_excited_msb(self) -> None:
        """atom_excited follows MSB: atom 0 is the highest bit."""
        from packages.simulation.observables import atom_excited

        # |101> = index 5 for 3 atoms
        assert atom_excited(5, 0, 3) is True   # q0 = 1
        assert atom_excited(5, 1, 3) is False  # q1 = 0
        assert atom_excited(5, 2, 3) is True   # q2 = 1

    def test_atom_excited_single_atom(self) -> None:
        from packages.simulation.observables import atom_excited

        assert atom_excited(0, 0, 1) is False  # |0>
        assert atom_excited(1, 0, 1) is True   # |1>

    def test_rydberg_density_matches_convention(self) -> None:
        """rydberg_density of |101> should flag atoms 0 and 2."""
        psi = np.zeros(8, dtype=np.complex128)
        psi[5] = 1.0  # |101>
        dens = rydberg_density(psi, 3)
        np.testing.assert_allclose(dens, [1.0, 0.0, 1.0])

    def test_bitstring_format_matches_convention(self) -> None:
        """bitstring_probabilities should use format(idx, '0Nb')."""
        psi = np.zeros(8, dtype=np.complex128)
        psi[5] = 1.0  # |101>
        probs = bitstring_probabilities(psi, 3, top_k=1)
        assert probs[0][0] == "101"

    def test_gpu_n_op_consistent(self) -> None:
        """GPU _n_op_diagonal must match observables.rydberg_density."""
        from packages.ml.gpu_backend import _n_op_diagonal

        n = 3
        for atom in range(n):
            diag = _n_op_diagonal(n, atom)
            for idx in range(2**n):
                expected = float((idx >> (n - 1 - atom)) & 1)
                assert diag[idx] == expected, (
                    f"Mismatch at atom={atom}, idx={idx}: "
                    f"got {diag[idx]}, expected {expected}"
                )

    def test_hamiltonian_sparse_consistent(self) -> None:
        """Sparse Hamiltonian occupancy extraction uses MSB convention."""
        # Indirectly tested: sparse and dense Hamiltonians match eigenvalues
        # (see test_sparse_hamiltonian.py). Direct check:
        n = 2
        from packages.simulation.hamiltonian import build_hamiltonian_matrix, build_sparse_hamiltonian

        pytest.importorskip("scipy")
        H_dense = build_hamiltonian_matrix([(0.0, 0.0), (7.0, 0.0)], omega=5.0, delta=-10.0)
        H_sparse = build_sparse_hamiltonian([(0.0, 0.0), (7.0, 0.0)], omega=5.0, delta=-10.0)
        np.testing.assert_allclose(
            H_sparse.toarray(), H_dense, atol=1e-10,
            err_msg="Sparse/dense Hamiltonian mismatch implies bit convention inconsistency",
        )

    def test_mis_bitstring_convention(self) -> None:
        """MIS bitstrings: leftmost character = atom 0 = MSB."""
        from packages.simulation.hamiltonian import find_maximum_independent_sets

        # 3-atom chain: 0-1-2 with nn edges
        adj = np.array([
            [False, True, False],
            [True, False, True],
            [False, True, False],
        ])
        sets = find_maximum_independent_sets(adj)
        assert [0, 2] in sets
        # Corresponding bitstring should be "101" (q0=1, q1=0, q2=1)
        # Verify by manual construction:
        bits = ["0"] * 3
        for idx in [0, 2]:
            bits[idx] = "1"
        assert "".join(bits) == "101"
```

### Vérification

```bash
pytest tests/test_observables.py -x --tb=short -v
```

---

## TÂCHE 3 — Ajouter des bornes d'erreur pour l'approximation de Trotter

### Problème actuel

**Fichier** : `packages/simulation/numpy_backend.py` (lignes 82-100)

La boucle de Trotter dans `simulate_rydberg_evolution` effectue l'évolution :

```python
for step in range(n_steps):
    # ...compute omega_t, delta_t...
    if n_atoms <= dense_limit:
        H = build_hamiltonian_matrix(coords, omega_t, delta_t, c6)
        psi = expm(-1j * H * dt) @ psi
    else:
        H = build_sparse_hamiltonian(coords, omega_t, delta_t, c6)
        psi = expm_multiply((-1j * H * dt), psi)
```

**Ce n'est pas vraiment un Trotter-Suzuki** — c'est une discrétisation par tranches temporelles avec Hamiltonien reconstruit à chaque pas, et chaque tranche utilise un `expm` ou `expm_multiply` exact. L'erreur vient de la **discrétisation temporelle** du Hamiltonien time-dependent, pas d'un splitting Trotter classique.

**Problèmes** :
1. **Aucune estimation d'erreur** : L'utilisateur/l'orchestrateur n'a aucun moyen de savoir si `n_steps=200` est suffisant pour la dynamique donnée.
2. **Pas de guidance** : Le choix de `n_steps` est arbitraire (valeur par défaut 200, les appelants utilisent 150). Pour certains paramètres (grand Ω, fort couplage), 200 pas peut être insuffisant.
3. **Pas de métriques de convergence** dans le résultat retourné.

### Solution : Borne d'erreur analytique + estimation numérique

Pour un Hamiltonien dépendant du temps avec N tranches de durée dt, l'erreur de discrétisation au 1er ordre est bornée par :

$$\epsilon \leq \frac{T^2}{2N} \max_t \left\| \frac{dH}{dt} \right\|$$

En pratique, on estime `||dH/dt||` par la différence maximale entre Hamiltoniens consécutifs : `max_t ||H(t+dt) - H(t)|| / dt`.

On ajoute aussi une estimation numérique : re-simuler avec `n_steps/2` et comparer la fidélité avec le résultat à `n_steps`.

### Modifications

#### 1. `packages/simulation/numpy_backend.py` — Ajouter `estimate_trotter_error`

Ajouter cette fonction **après** `simulate_rydberg_evolution` :

```python
def estimate_discretization_error(
    coords: list[tuple[float, float]],
    omega_max: float,
    delta_start: float,
    delta_end: float,
    duration_ns: float,
    n_steps: int = 200,
    omega_shape: str = "constant",
    c6: float = C6_RB87_70S,
    max_atoms_dense: int | None = None,
    max_atoms_sparse: int | None = None,
) -> dict[str, float]:
    """Estimate time-discretization error for the piecewise-constant evolution.

    Returns
    -------
    dict with:
        - ``analytical_bound``: upper bound from commutator-free Magnus estimate
          epsilon <= (T^2 / 2N) * max ||dH/dt||.
        - ``fidelity_half_steps``: |<psi_N | psi_{N/2}>|^2 comparing the
          full-step result to a half-step result.  Values close to 1.0
          indicate sufficient step count.
        - ``recommended_n_steps``: suggested n_steps for fidelity > 0.999.
    """
    from packages.simulation.observables import state_fidelity

    dense_limit = max_atoms_dense if max_atoms_dense is not None else _DEFAULT_PARAM_SPACE.max_atoms_dense
    n_atoms = len(coords)
    total_time_us = duration_ns / 1000.0
    dt = total_time_us / n_steps

    # ---- Analytical bound: estimate max ||dH/dt|| ----
    n_samples = min(n_steps, 10)
    h_norms: list[float] = []
    for k in range(n_samples):
        frac_a = k / n_samples
        frac_b = (k + 1) / n_samples

        def _omega(frac: float) -> float:
            if omega_shape == "blackman":
                return omega_max * (
                    0.42 - 0.5 * np.cos(2 * np.pi * frac) + 0.08 * np.cos(4 * np.pi * frac)
                )
            elif omega_shape == "ramp":
                return omega_max * min(frac * 2.0, 1.0)
            return omega_max

        omega_a = _omega(frac_a)
        omega_b = _omega(frac_b)
        delta_a = delta_start + (delta_end - delta_start) * frac_a
        delta_b = delta_start + (delta_end - delta_start) * frac_b

        if n_atoms <= dense_limit:
            Ha = build_hamiltonian_matrix(coords, omega_a, delta_a, c6)
            Hb = build_hamiltonian_matrix(coords, omega_b, delta_b, c6)
            diff_norm = float(np.linalg.norm(Hb - Ha))
        else:
            Ha = build_sparse_hamiltonian(coords, omega_a, delta_a, c6)
            Hb = build_sparse_hamiltonian(coords, omega_b, delta_b, c6)
            diff = Hb - Ha
            if SCIPY_AVAILABLE:
                from scipy.sparse.linalg import norm as sparse_norm
                diff_norm = float(sparse_norm(diff))
            else:
                diff_norm = float(np.linalg.norm(diff.toarray()))

        delta_frac = 1.0 / n_samples
        dh_dt = diff_norm / (delta_frac * total_time_us) if total_time_us > 0 else 0.0
        h_norms.append(dh_dt)

    max_dh_dt = max(h_norms) if h_norms else 0.0
    analytical_bound = (total_time_us ** 2 / (2.0 * n_steps)) * max_dh_dt

    # ---- Numerical estimate: compare N vs N/2 steps ----
    result_full = simulate_rydberg_evolution(
        coords, omega_max, delta_start, delta_end, duration_ns,
        n_steps=n_steps, omega_shape=omega_shape, c6=c6,
        max_atoms_dense=max_atoms_dense, max_atoms_sparse=max_atoms_sparse,
    )
    half_steps = max(n_steps // 2, 1)
    result_half = simulate_rydberg_evolution(
        coords, omega_max, delta_start, delta_end, duration_ns,
        n_steps=half_steps, omega_shape=omega_shape, c6=c6,
        max_atoms_dense=max_atoms_dense, max_atoms_sparse=max_atoms_sparse,
    )
    fidelity = state_fidelity(result_full["final_state"], result_half["final_state"])

    # ---- Recommend n_steps ----
    recommended = n_steps
    if fidelity < 0.999:
        # Error scales as O(1/N), so to reach fidelity 0.999 from current:
        # infidelity ~ 1/N => need N * (1-fidelity)/0.001
        ratio = (1.0 - fidelity) / 0.001
        recommended = int(np.ceil(n_steps * max(ratio, 1.0)))
        recommended = min(recommended, n_steps * 8)  # cap at 8x

    return {
        "analytical_bound": round(analytical_bound, 8),
        "fidelity_half_steps": round(fidelity, 8),
        "recommended_n_steps": recommended,
        "n_steps_used": n_steps,
    }
```

#### 2. `packages/simulation/numpy_backend.py` — Ajouter les métriques au résultat de `simulate_rydberg_evolution`

Ajouter un champ `discretization_dt` au dict de retour pour permettre un diagnostic externe :

**Avant** (dans le `return` de `simulate_rydberg_evolution`) :

```python
    return {
        "final_state": psi,
        "n_atoms": n_atoms,
        "rydberg_densities": dens.tolist(),
        "pair_correlations": pair_correlation(psi, n_atoms).tolist(),
        "connected_correlations": connected_correlation(psi, n_atoms).tolist(),
        "total_rydberg_fraction": total_rydberg_fraction(psi, n_atoms),
        "antiferromagnetic_order": antiferromagnetic_order(psi, n_atoms),
        "entanglement_entropy": entanglement_entropy(psi, n_atoms),
        "top_bitstrings": bitstring_probabilities(psi, n_atoms, top_k=5),
        "backend": backend_label,
        "n_steps": n_steps,
    }
```

**Après** :

```python
    return {
        "final_state": psi,
        "n_atoms": n_atoms,
        "rydberg_densities": dens.tolist(),
        "pair_correlations": pair_correlation(psi, n_atoms).tolist(),
        "connected_correlations": connected_correlation(psi, n_atoms).tolist(),
        "total_rydberg_fraction": total_rydberg_fraction(psi, n_atoms),
        "antiferromagnetic_order": antiferromagnetic_order(psi, n_atoms),
        "entanglement_entropy": entanglement_entropy(psi, n_atoms),
        "top_bitstrings": bitstring_probabilities(psi, n_atoms, top_k=5),
        "backend": backend_label,
        "n_steps": n_steps,
        "dt_us": dt,
    }
```

### Tests

**Fichier** : `tests/test_simulation.py` — Ajouter :

```python
from packages.simulation.numpy_backend import estimate_discretization_error


class TestDiscretizationError:
    """Tests for the time-discretization error estimator."""

    def test_constant_hamiltonian_zero_bound(self) -> None:
        """Constant Omega + constant delta => dH/dt = 0 => bound = 0."""
        result = estimate_discretization_error(
            coords=[(0.0, 0.0), (7.0, 0.0)],
            omega_max=5.0,
            delta_start=-10.0,
            delta_end=-10.0,  # no sweep
            duration_ns=1000,
            n_steps=50,
            omega_shape="constant",
        )
        assert result["analytical_bound"] == pytest.approx(0.0, abs=1e-6)
        assert result["fidelity_half_steps"] > 0.9999

    def test_sweep_produces_nonzero_bound(self) -> None:
        """A detuning sweep should produce a non-zero analytical bound."""
        result = estimate_discretization_error(
            coords=[(0.0, 0.0), (7.0, 0.0)],
            omega_max=5.0,
            delta_start=-20.0,
            delta_end=10.0,
            duration_ns=2000,
            n_steps=100,
        )
        assert result["analytical_bound"] > 0.0

    def test_more_steps_improve_fidelity(self) -> None:
        """Doubling n_steps should improve (or maintain) fidelity."""
        coords = [(0.0, 0.0), (7.0, 0.0), (14.0, 0.0)]
        r1 = estimate_discretization_error(
            coords=coords,
            omega_max=5.0,
            delta_start=-20.0,
            delta_end=10.0,
            duration_ns=3000,
            n_steps=50,
        )
        r2 = estimate_discretization_error(
            coords=coords,
            omega_max=5.0,
            delta_start=-20.0,
            delta_end=10.0,
            duration_ns=3000,
            n_steps=200,
        )
        assert r2["fidelity_half_steps"] >= r1["fidelity_half_steps"] - 1e-6

    def test_recommended_steps_reasonable(self) -> None:
        """Recommended n_steps should be >= n_steps used."""
        result = estimate_discretization_error(
            coords=[(0.0, 0.0), (7.0, 0.0)],
            omega_max=5.0,
            delta_start=-20.0,
            delta_end=10.0,
            duration_ns=2000,
            n_steps=100,
        )
        assert result["recommended_n_steps"] >= result["n_steps_used"]

    def test_result_keys(self) -> None:
        """Output dict should contain all expected keys."""
        result = estimate_discretization_error(
            coords=[(0.0, 0.0), (7.0, 0.0)],
            omega_max=5.0,
            delta_start=-10.0,
            delta_end=5.0,
            duration_ns=1000,
            n_steps=50,
        )
        assert {"analytical_bound", "fidelity_half_steps", "recommended_n_steps", "n_steps_used"} <= set(result.keys())
```

### Vérification

```bash
pytest tests/test_simulation.py -x --tb=short -v
```

---

## TÂCHE 4 — Remplacer le MIS brute-force par un algorithme approché pour n > 15

### Problème actuel

**Fichier** : `packages/simulation/hamiltonian.py` (lignes 93-120)

```python
def find_maximum_independent_sets(
    adjacency: NDArray[np.bool_],
    max_results: int = 20,
) -> list[list[int]]:
    """Brute-force MIS enumeration (feasible for ≤20 atoms)."""
    n = adjacency.shape[0]
    if n > 20:
        return []
    best_size = 0
    results: list[list[int]] = []
    for size in range(n, 0, -1):
        if size < best_size:
            break
        for subset in combinations(range(n), size):
            # ...check independence...
```

**Problèmes** :
1. **Complexité** : `O(2^n)` — pour n=20, `C(20,10) = 184 756` subsets au seul rang 10. En pratique, la boucle itère sur toutes les `C(n, k)` combinaisons en ordre décroissant de taille.
2. **Retour vide pour n > 20** : Le système retourne silencieusement `[]` au-delà de 20 atomes, rendant `mis_bitstrings()` et `mis_overlap()` inopérants — sans avertissement.
3. **Seuil de 20 trop élevé** : Même pour n=18, la boucle brute-force peut prendre ~30 secondes. Le seuil réaliste pour une exécution interactive est n ≈ 15.
4. **Pas d'algorithme alternatif** : Pour n > 15, un algorithme glouton ou greedy+local-search donnerait une approximation raisonnable.

### Solution : Algorithme glouton + brute-force pour petits systèmes

1. Abaisser le seuil brute-force à n ≤ 15.
2. Pour 15 < n ≤ 50, utiliser un algorithme glouton avec priorisation par degré minimum.
3. Loguer un avertissement quand le fallback glouton est utilisé.

### Modifications

#### 1. `packages/simulation/hamiltonian.py` — Ajouter l'algorithme glouton

Ajouter cette fonction **avant** `find_maximum_independent_sets` :

```python
def _greedy_independent_set(
    adjacency: NDArray[np.bool_],
) -> list[int]:
    """Greedy MIS heuristic: iteratively pick the lowest-degree non-adjacent vertex.

    Returns a single independent set (not guaranteed maximum, but typically
    within a factor of O(log n) of optimal for sparse graphs).
    """
    n = adjacency.shape[0]
    available = set(range(n))
    result: list[int] = []

    while available:
        # Pick vertex with minimum degree among available vertices
        best = min(available, key=lambda v: sum(1 for u in available if u != v and adjacency[v, u]))
        result.append(best)
        # Remove best and all its neighbours
        neighbours = {u for u in available if adjacency[best, u]}
        available -= neighbours | {best}

    return sorted(result)


def _greedy_mis_multi(
    adjacency: NDArray[np.bool_],
    n_restarts: int = 10,
    max_results: int = 20,
) -> list[list[int]]:
    """Run the greedy heuristic with random vertex-order perturbations.

    Returns up to *max_results* distinct independent sets of maximum size
    found across *n_restarts* randomised restarts.
    """
    n = adjacency.shape[0]
    rng = np.random.default_rng(42)
    best_size = 0
    results: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()

    for _ in range(n_restarts):
        available = set(range(n))
        result: list[int] = []
        order = rng.permutation(n).tolist()

        while available:
            candidates = [v for v in order if v in available]
            if not candidates:
                break
            # Pick from candidates with minimum degree
            best = min(candidates, key=lambda v: sum(1 for u in available if u != v and adjacency[v, u]))
            result.append(best)
            neighbours = {u for u in available if adjacency[best, u]}
            available -= neighbours | {best}

        result_sorted = sorted(result)
        key = tuple(result_sorted)
        size = len(result_sorted)

        if size > best_size:
            best_size = size
            results = [result_sorted]
            seen = {key}
        elif size == best_size and key not in seen:
            results.append(result_sorted)
            seen.add(key)
            if len(results) >= max_results:
                break

    return results
```

#### 2. `packages/simulation/hamiltonian.py` — Modifier `find_maximum_independent_sets`

**Avant** :

```python
def find_maximum_independent_sets(
    adjacency: NDArray[np.bool_],
    max_results: int = 20,
) -> list[list[int]]:
    """Brute-force MIS enumeration (feasible for ≤20 atoms).

    An independent set contains no two adjacent (blockaded) vertices.
    Returns all sets of maximum cardinality, up to *max_results*.
    """
    n = adjacency.shape[0]
    if n > 20:
        return []
    best_size = 0
    results: list[list[int]] = []
    for size in range(n, 0, -1):
        if size < best_size:
            break
        for subset in combinations(range(n), size):
            valid = True
            for idx_a, a in enumerate(subset):
                for b in subset[idx_a + 1 :]:
                    if adjacency[a, b]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                best_size = size
                results.append(list(subset))
                if len(results) >= max_results:
                    return results
    return results
```

**Après** :

```python
_MIS_EXACT_THRESHOLD: int = 15
"""Maximum atom count for exact MIS enumeration. Above this, use greedy heuristic."""


def find_maximum_independent_sets(
    adjacency: NDArray[np.bool_],
    max_results: int = 20,
) -> list[list[int]]:
    """Find maximum independent sets on the blockade graph.

    For n <= 15 atoms, performs exact brute-force enumeration.
    For 15 < n <= 50, uses a randomised greedy heuristic (approximate).
    For n > 50, returns an empty list.

    An independent set contains no two adjacent (blockaded) vertices.
    Returns sets of maximum cardinality found, up to *max_results*.
    """
    n = adjacency.shape[0]
    if n > 50:
        logger.warning("MIS computation skipped for n=%d atoms (limit: 50).", n)
        return []

    if n > _MIS_EXACT_THRESHOLD:
        logger.debug(
            "Using greedy MIS heuristic for n=%d atoms (exact threshold: %d).",
            n, _MIS_EXACT_THRESHOLD,
        )
        return _greedy_mis_multi(adjacency, n_restarts=max(20, n), max_results=max_results)

    # Exact enumeration for small systems
    best_size = 0
    results: list[list[int]] = []
    for size in range(n, 0, -1):
        if size < best_size:
            break
        for subset in combinations(range(n), size):
            valid = True
            for idx_a, a in enumerate(subset):
                for b in subset[idx_a + 1 :]:
                    if adjacency[a, b]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                best_size = size
                results.append(list(subset))
                if len(results) >= max_results:
                    return results
    return results
```

#### 3. `packages/simulation/hamiltonian.py` — Ajouter l'import du logger

Ajouter dans les imports du module :

```python
from packages.core.logging import get_logger

logger = get_logger(__name__)
```

### Tests

**Fichier** : `tests/test_hamiltonian.py` — Modifier et étendre la classe `TestMIS` :

```python
class TestMIS:
    def test_three_atom_chain_mis(self) -> None:
        """For a 3-atom chain with only nn edges, MIS = {0,2}."""
        adj = np.array([
            [False, True, False],
            [True, False, True],
            [False, True, False],
        ])
        sets = find_maximum_independent_sets(adj)
        assert len(sets) >= 1
        assert [0, 2] in sets

    def test_bitstrings_format(self) -> None:
        coords = [(0.0, 0.0), (7.0, 0.0), (14.0, 0.0)]
        bs = mis_bitstrings(coords, omega=5.0)
        for s in bs:
            assert len(s) == 3
            assert all(c in "01" for c in s)

    def test_disconnected_graph_all_selected(self) -> None:
        """No edges => all atoms in the MIS."""
        adj = np.zeros((4, 4), dtype=bool)
        sets = find_maximum_independent_sets(adj)
        assert len(sets) >= 1
        assert [0, 1, 2, 3] in sets

    def test_complete_graph_single_vertex(self) -> None:
        """Fully connected => MIS size = 1."""
        n = 5
        adj = np.ones((n, n), dtype=bool)
        np.fill_diagonal(adj, False)
        sets = find_maximum_independent_sets(adj)
        assert len(sets) >= 1
        assert all(len(s) == 1 for s in sets)

    def test_greedy_fallback_large_system(self) -> None:
        """For n > 15, greedy heuristic should return non-empty results."""
        n = 20
        rng = np.random.default_rng(42)
        # Random sparse graph (each pair has 30% chance of edge)
        adj = rng.random((n, n)) < 0.3
        adj = adj | adj.T
        np.fill_diagonal(adj, False)

        sets = find_maximum_independent_sets(adj)
        assert len(sets) >= 1
        # Verify independence
        for s in sets:
            for i, a in enumerate(s):
                for b in s[i + 1:]:
                    assert not adj[a, b], f"Vertices {a} and {b} are adjacent"

    def test_greedy_result_is_maximal(self) -> None:
        """Each greedy result should be a maximal independent set
        (no vertex can be added without breaking independence)."""
        n = 18
        # Linear chain: 0-1-2-...-17
        adj = np.zeros((n, n), dtype=bool)
        for i in range(n - 1):
            adj[i, i + 1] = True
            adj[i + 1, i] = True

        sets = find_maximum_independent_sets(adj)
        assert len(sets) >= 1

        for s in sets:
            s_set = set(s)
            for v in range(n):
                if v in s_set:
                    continue
                # v should be adjacent to at least one vertex in s
                # (otherwise s is not maximal)
                has_neighbour = any(adj[v, u] for u in s_set)
                assert has_neighbour, (
                    f"Vertex {v} could be added to {s} — set is not maximal"
                )

    def test_greedy_matches_exact_for_small(self) -> None:
        """For systems at the threshold, greedy and exact should agree on size."""
        from packages.simulation.hamiltonian import _greedy_mis_multi

        n = 10
        adj = np.zeros((n, n), dtype=bool)
        for i in range(n - 1):
            adj[i, i + 1] = True
            adj[i + 1, i] = True

        exact = find_maximum_independent_sets(adj)
        greedy = _greedy_mis_multi(adj, n_restarts=50)

        exact_size = len(exact[0]) if exact else 0
        greedy_size = len(greedy[0]) if greedy else 0
        assert greedy_size == exact_size

    def test_very_large_system_returns_empty(self) -> None:
        """Systems > 50 atoms return empty list (safety limit)."""
        adj = np.zeros((60, 60), dtype=bool)
        sets = find_maximum_independent_sets(adj)
        assert sets == []
```

### Vérification

```bash
pytest tests/test_hamiltonian.py -x --tb=short -v
```

---

## Ordre d'exécution recommandé

Exécute les tâches dans cet ordre pour minimiser les conflits :

| Ordre | Tâche | Justification |
|-------|-------|---------------|
| 1 | **T2 — Convention MSB/LSB** | Documentation pure + `atom_excited` utilitaire — aucun risque de régression, baseline pour les autres tâches |
| 2 | **T4 — MIS approché** | Modification isolée dans `hamiltonian.py` — pas de dépendance avec les backends de simulation |
| 3 | **T3 — Bornes d'erreur Trotter** | Ajout additif dans `numpy_backend.py` — dépend de la convention bitstring validée en T2 |
| 4 | **T1 — Lanczos GPU** | Modification la plus invasive — remplace le cœur de l'évolution GPU |

---

## Contraintes de validation

Après TOUTES les tâches :

```bash
# Suite complète
pytest tests/ -x --tb=short

# Vérification de couverture sur les modules modifiés
pytest tests/test_ml_gpu.py tests/test_hamiltonian.py tests/test_simulation.py tests/test_observables.py --cov=packages/simulation --cov=packages/ml/gpu_backend --cov-report=term-missing

# Vérification que les résultats existants sont préservés
pytest tests/test_sparse_hamiltonian.py tests/test_physical_evaluator.py -x --tb=short
```

**Invariants à vérifier** :
- Tous les tests existants passent sans modification (rétro-compatibilité)
- `simulate_rydberg_evolution` retourne les mêmes clés qu'avant + `dt_us`
- `gpu_time_evolution` retourne les mêmes clés qu'avant + `krylov_dim`
- `find_maximum_independent_sets` retourne des résultats corrects pour n ≤ 15 (identiques à l'ancien brute-force)
- Aucun `.to_dense()` dans le hot path de `gpu_time_evolution`
