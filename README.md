# CryoSwarm-Q

**Orchestration multi-agent pour la conception autonome d'expériences sur atomes neutres.**

CryoSwarm-Q, c'est la couche logicielle qui se place entre l'intention du chercheur et la programmation pulse-level sur processeurs à atomes neutres. Tu lui donnes un objectif expérimental structuré, il génère des registres atomiques candidats, des séquences de pulses, il évalue tout ça sous bruit réaliste, il classe par robustesse et faisabilité, et il retient les leçons pour la prochaine campagne.

Le système cible les processeurs Pasqal à atomes de $^{87}\text{Rb}$ piégés optiquement et excités vers des états de Rydberg. Toute l'évaluation repose sur le Hamiltonien Rydberg driven — pas de modèles jouets, pas d'approximations vagues. Schrödinger n'a pas mis son chat dans une boîte pour qu'on fasse semblant.

---

## Sommaire

- [Pourquoi CryoSwarm-Q](#pourquoi-cryoswarm-q) — *Le problème avant la solution*
- [Fondations mathématiques](#fondations-mathématiques) — *Les équations qu'on respecte*
- [Architecture multi-agent](#architecture-multi-agent) — *La chaîne de commandement*
- [Pipeline ML](#pipeline-ml) — *Quand le réseau apprend le métier*
- [Dashboard interactif](#dashboard-interactif-streamlit) — *Le centre de contrôle*
- [API REST](#api-rest) — *Pour ceux qui préfèrent les endpoints*
- [Installation](#installation) — *Trois commandes et c'est parti*
- [Démarrage rapide](#démarrage-rapide) — *Du concret, vite*
- [Usage programmatique](#usage-programmatique) — *Pour les scripteurs*
- [Suite de tests](#suite-de-tests) — *372 tests, zéro regret*
- [Cartographie du repo](#cartographie-du-repo) — *Plan de la base*
- [Limitations actuelles](#limitations-actuelles) — *Ce qu'on ne prétend pas faire*
- [Licence](#licence)

---

## Pourquoi CryoSwarm-Q

Designer une expérience sur plateforme à atomes neutres, c'est un peu comme Boltzmann face à l'entropie : le problème est clair, mais l'espace des solutions est immense. Le placement des atomes est programmable, le contrôle des pulses est flexible, et les contraintes hardware sont strictes (rayon de blocage, bornes sur la fréquence de Rabi, géométrie du device). Ajoutons le bruit — fluctuations laser, déphasage, erreurs SPAM, désordre de position — et on obtient un problème d'optimisation que personne ne résout à la main au-delà de quelques atomes.

CryoSwarm-Q automatise la boucle complète de conception :

1. **Cadrage du problème** — traduire un objectif scientifique en spécification d'expérience concrète.
2. **Génération de géométrie** — proposer des registres atomiques qui respectent les contraintes hardware (espacement minimal, conditions de blocage, bornes du device).
3. **Design de séquences de pulses** — générer des candidats sur 5 familles de formes d'onde, par heuristiques, apprentissage par renforcement ou stratégies hybrides.
4. **Évaluation sous bruit** — simuler chaque candidat en conditions nominales et perturbées (bruit d'amplitude, dérive du désaccord, déphasage, perte d'atomes, SPAM, inhomogénéité spatiale).
5. **Classement par robustesse** — scorer et classer les candidats selon un objectif composite qui balance alignement de l'observable, robustesse, coût d'exécution et latence.
6. **Mémoire** — stocker les leçons apprises pour que les campagnes futures démarrent avec du contexte.

L'objectif n'est pas de remplacer le physicien — c'est de systématiser l'exploration de l'espace de design, de faire remonter les compromis non évidents, et de fournir un raisonnement transparent et reproductible à chaque étape. Comme disait Feynman : "What I cannot create, I do not understand." Ici, on crée les candidats ET on les comprend.

---

## Fondations mathématiques

*Toute la physique du projet tient dans ces équations. On ne triche pas, on ne simplifie pas à outrance. Dirac serait fier — ou au moins pas mécontent.*

### Le Hamiltonien Rydberg

Le cœur de CryoSwarm-Q, c'est le Hamiltonien driven Rydberg pour $N$ atomes. Tout part de là :

$$
\hat{H} = \frac{\Omega}{2}\sum_{i=1}^{N} \hat{\sigma}_x^{(i)} - \delta\sum_{i=1}^{N} \hat{n}_i + \sum_{i<j} \frac{C_6}{|\mathbf{r}_i - \mathbf{r}_j|^6}\,\hat{n}_i\hat{n}_j
$$

avec :

| Symbole | Signification | Plage typique |
|---------|---------------|---------------|
| $\Omega$ | Fréquence de Rabi (amplitude du drive) | 1 – 15 rad/μs |
| $\delta$ | Désaccord (offset énergétique de l'état Rydberg) | −40 à +25 rad/μs |
| $\hat{n}_i = \|r_i\rangle\langle r_i\|$ | Opérateur nombre Rydberg pour l'atome $i$ | — |
| $C_6$ | Coefficient de Van der Waals | $2\pi \times 862\,690$ rad·μm⁶/μs |
| $\mathbf{r}_i$ | Position de l'atome $i$ dans le réseau de pinces | Espacement : 4 – 15 μm |

L'espèce atomique est le $^{87}\text{Rb}$ dans l'état Rydberg $|70S_{1/2}\rangle$.

### Rayon de blocage

Quand deux atomes sont plus proches que le rayon de blocage $R_b$, l'excitation Rydberg simultanée est énergétiquement interdite — comme deux rois sur un échiquier, ils ne peuvent pas coexister trop près :

$$
R_b = \left(\frac{C_6}{\Omega}\right)^{1/6}
$$

À $\Omega = 5$ rad/μs, on obtient $R_b \approx 9.8$ μm. Ce blocage est le pont entre la physique Rydberg et l'optimisation combinatoire (MIS).

### Maximum Independent Set (MIS)

L'état fondamental du Hamiltonien Rydberg en régime de blocage (grand $\delta$ positif, interactions fortes) approxime les solutions du problème Maximum Independent Set sur le graphe de blocage. En gros, Maxwell résolvait les champs, nous on résout des graphes avec des atomes :

$$
\text{MIS}(G) = \arg\max_{S \subseteq V}\; |S| \quad \text{s.t.} \quad \forall\,(u,v)\in E,\; u \notin S \;\text{or}\; v \notin S
$$

CryoSwarm-Q calcule le MIS exact pour $N \leq 15$ atomes (énumération brute-force) et utilise des heuristiques gloutones avec redémarrages aléatoires pour les systèmes plus grands.

### Observables quantiques

La couche de simulation calcule ces observables à partir de la fonction d'onde de l'état fondamental $|\psi\rangle$ :

| Observable | Formule | Signification physique |
|-----------|---------|------------------------|
| Densité Rydberg | $\langle \hat{n}_i \rangle$ | Probabilité d'excitation de l'atome $i$ |
| Fraction Rydberg totale | $\bar{n} = \frac{1}{N}\sum_i \langle \hat{n}_i \rangle$ | Excitation moyenne sur le registre |
| Corrélation connectée | $g_{ij}^{(2)} = \langle \hat{n}_i \hat{n}_j \rangle - \langle \hat{n}_i \rangle \langle \hat{n}_j \rangle$ | Corrélations quantiques au-delà du champ moyen |
| Ordre antiferromagnétique | $m_{\text{AF}} = \frac{1}{N}\left\|\sum_i (-1)^i (2\langle \hat{n}_i \rangle - 1)\right\|$ | Magnétisation alternée (1 = ordre de Néel parfait) |
| Entropie d'intrication | $S_A = -\text{Tr}(\rho_A \log_2 \rho_A)$ | Intrication bipartite via décomposition de Schmidt |
| Probabilité de bitstring | $P(b) = \|\langle b \| \psi \rangle\|^2$ | Probabilité de mesurer chaque état de la base computationnelle |

### Familles de séquences de pulses

CryoSwarm-Q génère des schedules de pulses à partir de 5 familles — chacune a sa personnalité, comme les 5 postulats de la mécanique quantique (sauf que ceux-là marchent du premier coup) :

| Famille | $\Omega(t)$ | $\delta(t)$ | Cas d'usage |
|---------|------------|------------|-------------|
| `constant_drive` | Constante $\Omega_{\max}$ | Constante $\delta_0$ | Oscillation de Rabi simple |
| `global_ramp` | Rampe linéaire $0 \to \Omega_{\max}$ | Balayage linéaire | Excitation progressive |
| `detuning_scan` | Constante $\Omega_{\max}$ | Balayage linéaire | Scan de résonance |
| `adiabatic_sweep` | Enveloppe $\sin^2(\pi t/T)$ | Balayage linéaire | Préparation d'état adiabatique |
| `blackman_sweep` | Fenêtre de Blackman | Balayage linéaire | Préparation adiabatique basse fuite |

L'évolution temporelle suit $|\psi(t)\rangle = \mathcal{T}\exp\left(-i\int_0^t \hat{H}(t')\,dt'\right)|\psi(0)\rangle$, discrétisée par décomposition de Trotter-Suzuki au second ordre.

### Scoring et classement

Chaque candidat reçoit un score objectif composite — parce qu'un seul nombre ne raconte jamais toute l'histoire (Heisenberg l'avait compris avant tout le monde) :

$$
S_{\text{obj}} = \alpha\, s_{\text{obs}} + \beta\, S_{\text{robust}} - \gamma\, c_{\text{exec}} - \delta_w\, \ell_{\text{latency}}
$$

avec les poids par défaut $\alpha = 0.45$, $\beta = 0.35$, $\gamma = 0.10$, $\delta_w = 0.10$.

Le scoring de robustesse agrège simulations nominales et perturbées :

$$
S_{\text{robust}} = w_n \, s_{\text{nom}} + w_a \, \bar{s}_{\text{pert}} + w_w \, s_{\text{worst}} + w_s \, b_{\text{stab}}
$$

avec les poids $w_n = 0.25$, $w_a = 0.35$, $w_w = 0.30$, $w_s = 0.10$.

Le bonus de stabilité récompense la faible variance entre les scénarios de bruit, tandis qu'une pénalité signale les candidats dont la performance chute brutalement entre le nominal et le pire cas.

### Modèle de bruit

Trois scénarios de perturbation sont appliqués à chaque candidat — parce qu'un bon candidat, c'est celui qui tient la route même quand Laplace arrête de croire au déterminisme :

| Scénario | $\sigma_{\text{amp}}$ | $\sigma_{\text{det}}$ | $\gamma_\phi$ | $\gamma_{\text{loss}}$ | $T$ (μK) | SPAM | Spatial |
|----------|----------------------|----------------------|---------------|----------------------|----------|------|---------|
| BAS | 0.01 | 0.01 | 0.001 | 0.001 | 50 | 0.5% | 2% |
| MOYEN | 0.05 | 0.05 | 0.005 | 0.005 | 50 | 0.5% | 5% |
| STRESSÉ | 0.10 | 0.10 | 0.010 | 0.010 | 50 | 0.5% | 8% |

Les canaux de bruit incluent : fluctuation d'amplitude ($\Omega \to \Omega(1+\epsilon_\Omega)$), dérive du désaccord, déphasage Lindblad, perte d'atomes, erreurs SPAM, et inhomogénéité spatiale du drive.

---

## Architecture multi-agent

*Comme disait von Neumann : "Avec quatre paramètres je peux modéliser un éléphant, avec cinq je le fais bouger la trompe." Ici on a 8 agents, et chacun sait exactement quoi faire.*

CryoSwarm-Q repose sur des agents spécialisés avec des responsabilités explicites, orchestrés dans un pipeline déterministe :

```
ObjectifExpérimental
      │
      ▼
┌─────────────────┐
│ ProblemFraming   │  Objectif → SpécificationExpérience
│     Agent        │  (nb atomes, géométrie, observable cible)
└────────┬────────┘
         ▼
┌─────────────────┐
│  GeometryAgent   │  Propose des registres faisables hardware
│                  │  (espacement, blocage, bornes du device)
└────────┬────────┘
         ▼
┌─────────────────┐
│ SequenceStrategy │  Génère les candidats pulse
│ (heuristique/RL/ │  (5 familles × variations de paramètres)
│  hybride/bandit) │
└────────┬────────┘
         ▼
┌─────────────────┐
│ SurrogateFilter  │  Pré-filtre les candidats (optionnel)
│                  │  (prédiction d'ensemble + incertitude)
└────────┬────────┘
         ▼
┌─────────────────┐
│ NoiseRobustness  │  Évalue nominal + 3 scénarios de bruit
│     Agent        │  (évaluation parallèle supportée)
└────────┬────────┘
         ▼
┌─────────────────┐
│ CampaignAgent    │  Classe les candidats par score composite
│                  │  (lexicographique : obj > worst > robust > nom)
└────────┬────────┘
         ▼
┌─────────────────┐
│  MemoryAgent     │  Extrait et stocke les leçons réutilisables
│                  │  (taggées par classe de problème)
└────────┬────────┘
         ▼
    RésuméPipeline
```

Chaque agent produit une `AgentDecision` avec sortie structurée, raisonnement et horodatage — traçabilité totale.

### Modes de stratégie de séquence

Le `SequenceStrategy` sélectionne comment les candidats pulse sont générés :

| Mode | Description |
|------|-------------|
| `heuristic_only` | Balayages de paramètres sur les 5 familles de formes d'onde |
| `rl_only` | La politique PPO propose les candidats directement |
| `hybrid` | Évalue les candidats heuristiques et RL ensemble |
| `adaptive` | Change de stratégie par classe de problème selon l'historique |
| `bandit` | Bandit multi-bras UCB1 sélectionne la meilleure stratégie |

---

## Pipeline ML

*Le ML ici, c'est pas du fine-tuning de LLM sur des tweets. C'est de l'apprentissage qui sert la physique — Planck aurait approuvé (après avoir râlé sur les constantes).*

### Ensemble de surrogates

Trois réseaux `SurrogateModelV2` entraînés indépendamment prédisent les scores de robustesse sans simulation complète — un raccourci honnête :

```
Input(18) → Linear(128) → GELU →
  [ResidualBlock(LayerNorm → Linear → GELU → Dropout → Linear)] × 3
→ LayerNorm → Linear(64) → GELU → Linear(4) → Sigmoid
```

L'entrée (18 dimensions) est un vecteur de features physiquement motivé (nombre d'atomes, ratios d'espacement, métriques de blocage, paramètres de pulse). Les 4 sorties sont `(robustness_score, nominal_score, worst_case_score, observable_score)`.

L'incertitude épistémique est estimée par la variance inter-modèles :

$$
\sigma_{\text{epist}}^2 = \frac{1}{M}\sum_{m=1}^{M} (f_m(\mathbf{x}) - \hat{y}_{\text{ens}})^2
$$

Les candidats à forte incertitude sont priorisés pour re-simulation (apprentissage actif).

### Apprentissage par renforcement PPO

Un réseau `ActorCritic` apprend à proposer des paramètres de pulse directement — pas d'heuristique, le réseau fait ses propres choix :

- **Observation** : vecteur 16-dim (nombre d'atomes, espacement, rayon de blocage, faisabilité, densité cible, meilleure robustesse jusqu'ici, meilleurs paramètres, progression de l'épisode)
- **Action** : 4-dim continu $[-1, 1]$ remis à l'échelle vers (amplitude, désaccord, durée, famille)
- **Objectif** : PPO clippé avec estimation d'avantage GAE

$$
L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}\left[\min\left(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta), 1 \pm \epsilon)\hat{A}_t\right)\right]
$$

### Boucle d'apprentissage actif

Le système itère entre entraînement du surrogate et entraînement RL — comme un chercheur qui alterne entre théorie et expérience, sauf qu'il ne dort jamais :

1. Entraîner / affiner l'ensemble surrogate sur les données de simulation
2. Entraîner PPO en utilisant les prédictions surrogate comme proxy de récompense rapide
3. Collecter les configurations RL les plus fortes
4. Sélectionner des points diversifiés à forte incertitude
5. Re-simuler avec l'évaluateur réel
6. Enrichir le dataset et recommencer

### Bandit de stratégie

Un bandit multi-bras UCB1 suit la performance par stratégie au fil des campagnes :

$$
\text{UCB1}(s) = \bar{r}_s + \sqrt{\frac{2\ln N}{n_s}}
$$

Ça balance automatiquement l'exploration de nouvelles stratégies et l'exploitation des bonnes connues.

---

## Dashboard interactif (Streamlit)

*8 pages, chacune dédiée à une étape du pipeline. Tu vois tout, tu comprends tout, tu peux tout inspecter. Galilée aurait adoré — lui qui insistait pour que tout soit observable.*

Lancement :

```bash
streamlit run apps/dashboard/app.py
```

### Page 1 — Centre de contrôle des campagnes

La page opérationnelle principale. Lance de nouvelles campagnes en spécifiant titre, objectif scientifique, nombre d'atomes (2–50), géométrie préférée, observable cible et priorité. Pendant l'exécution, un bus d'événements propage les décisions agents en temps réel.

Après exécution :
- **Campagnes récentes** — tableau des 20 dernières campagnes
- **Inspecteur de pipeline** — indicateurs de phase par agent + diagramme de Gantt
- **Entonnoir de candidats** — visualisation (Registres → Séquences → Évalués → Classés)
- **Log de décisions agents** — sorties structurées dépliables pour chaque appel

### Page 2 — Labo physique des registres

Exploration de la physique des géométries de registre :
- **Scatter 2D** des positions atomiques avec cercles de rayon de blocage (colorés par densité Rydberg quand disponible)
- **Heatmap d'interactions Van der Waals** — interactions $C_6 / r_{ij}^6$ par paires
- **Graphe de blocage** avec les MIS surlignés
- **Histogramme des distances** avec le seuil de blocage marqué
- **Slider $\Omega$ interactif** qui recalcule $R_b$ dynamiquement
- Métriques : nombre d'atomes, distance minimale, paires en blocage, score de faisabilité

### Page 3 — Spectroscopie Hamiltonienne

Diagonalisation en temps réel et analyse spectrale (jusqu'à 14 atomes) :
- **Spectre d'énergie** des 20 premiers niveaux
- **Distribution de probabilité de bitstring** avec les bitstrings MIS en or
- **Gap spectral** ($\Delta E = E_1 - E_0$), dimension de Hilbert ($2^N$), Inverse Participation Ratio
- Composition d'état : fraction Rydberg, entropie d'intrication, paramètre d'ordre AF
- **Balayage paramétrique en désaccord** révélant croisements évités et transitions de phase (pour $N \leq 10$)
- Chargement auto des contrôles depuis le meilleur candidat

### Page 4 — Studio de séquences de pulses

Inspection et design des formes d'onde :
- **Visualisation des formes d'onde** ($\Omega(t)$ et $\delta(t)$ dans le temps) pour chaque séquence candidate
- **Panneau mathématiques** — formules Blackman, balayage linéaire, Trotter-Suzuki
- **Table de comparaison** entre familles, amplitudes, désaccords, durées et coûts prédits
- **Scatter de l'espace des paramètres** (séquences colorées par score objectif)
- **Générateur de formes d'onde** — design libre (famille, $\Omega_{\max}$, $\delta_{\text{start}}$, $\delta_{\text{end}}$, durée)

### Page 5 — Arène de robustesse

Analyse de sensibilité au bruit et comparaison des scores — le stress-test des candidats, comme un concours de prépa mais pour des atomes :
- **Graphe barres groupées** — scores nominal, moyenne des perturbations et pire cas
- **Radar chart de bruit** — comparaison multi-candidats (jusqu'à 4)
- **Violin plot de robustesse** pour la distribution des scores
- **Deep dive** par candidat : cascade de dégradation + table d'observables (nominal vs BAS/MOYEN/STRESSÉ)
- Métriques : score nominal, moyenne perturbée, pire cas, écart-type, pénalité de robustesse
- Export du rapport complet (JSON)

### Page 6 — Observatoire ML

Monitoring du sous-système ML en 4 onglets :
- **Modèle Surrogate** — courbes de loss, résumé d'architecture (18→128→GELU→3 blocs résiduels→4), statut des checkpoints
- **Entraînement PPO** — courbes de récompense, loss policy, loss valeur, entropie, hyperparamètres, description des espaces action/observation
- **Bandit de stratégie** — évolution UCB1, pie chart de distribution, statistiques par stratégie
- **RL vs Heuristique** — barres groupées RL vs heuristique par trial

### Page 7 — Analytics de campagnes

Analyse de tendances inter-campagnes :
- **Timeline** (jusqu'à 50 campagnes)
- **Évolution des scores** en ligne
- **Distribution des backends** en barres empilées
- **Explorateur 3D** de l'espace des paramètres avec front de Pareto
- **Système de mémoire** — nuage de tags, filtres par type de leçon/campagne/tag, JSON dépliable
- Stats : total campagnes, candidats moyens par campagne, meilleur score historique, backend favori, taux de succès

### Page 8 — Référence théorique

Référence mathématique autonome en 8 sections dépliables :
1. **Hamiltonien Rydberg** — Hamiltonien complet, gap spectral, IPR, condition adiabatique
2. **Observables quantiques** — densité Rydberg, corrélations, ordre AF, entropie d'intrication, fidélité
3. **Maximum Independent Set** — définition MIS, mesure d'overlap, fonction de coût
4. **Familles de séquences de pulses** — 5 familles avec table de profils
5. **Scoring de robustesse** — formule, poids, bonus stabilité, pénalité, paramètres de bruit
6. **Score objectif et classement** — scoring composite, clé de classement lexicographique
7. **PPO et Apprentissage par renforcement** — objectif PPO, ratio d'importance, GAE, détails de l'environnement
8. **Ensemble surrogate et incertitude** — architecture, incertitude épistémique, stratégie UCB1

Chaque section renvoie vers le fichier d'implémentation correspondant.

---

## API REST

CryoSwarm-Q expose un backend FastAPI pour l'accès programmatique — parce que les vrais utilisent des endpoints, pas des boutons :

```bash
uvicorn apps.api.main:app --reload
```

URL de base : `/api/v1/`

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/api/v1/health` | GET | Health check et statut de connectivité MongoDB |
| `/api/v1/campaigns/` | POST | Lancer une nouvelle campagne depuis un objectif |
| `/api/v1/campaigns/` | GET | Lister les campagnes (avec pagination) |
| `/api/v1/campaigns/{id}` | GET | Détail complet d'une campagne |
| `/api/v1/candidates/{campaign_id}` | GET | Candidats classés pour une campagne |

---

## Installation

### Prérequis

- Python 3.10+
- MongoDB (optionnel — le système tombe automatiquement sur du stockage en mémoire)

### Installation éditable

```bash
pip install -e ".[dev]"
```

Ou installer les dépendances directement :

```bash
pip install -r requirements.txt
```

### Intégrations optionnelles

| Package | Rôle |
|---------|------|
| `pulser`, `pulser-simulation` | Construction native de séquences de pulses Pasqal |
| `pasqal-cloud` | Adaptateur d'exécution cloud |
| `qoolqit` | Toolkit d'optimisation quantique |
| `torch` | Modules ML (surrogate, PPO, apprentissage actif) |

---

## Démarrage rapide

### Lancer les tests

```bash
pytest tests -q
```

### Lancer le dashboard

```bash
streamlit run apps/dashboard/app.py
```

### Lancer l'API

```bash
uvicorn apps.api.main:app --reload
```

### Lancer une campagne démo (script)

```bash
python -m scripts.run_demo_pipeline
```

### Commandes d'entraînement ML

Générer un dataset d'entraînement :
```bash
python -m scripts.train_ml --phase generate_v2 --n-samples 1000 --workers 2 --sampling lhs
```

Entraîner l'ensemble surrogate :
```bash
python -m scripts.train_ml --phase surrogate --data data/generated/dataset.npz --epochs 100
```

Entraîner PPO :
```bash
python -m scripts.train_ml --phase rl --updates 500
```

Entraînement complet (surrogate + PPO) :
```bash
python -m scripts.train_ml --phase full --data data/generated/dataset.npz --epochs 100 --updates 500
```

Boucle d'apprentissage actif :
```bash
python -m scripts.train_ml --phase active --data data/generated/dataset.npz --al-iterations 5 --al-top-k 200
```

Benchmarks et ablations :
```bash
python -m scripts.benchmark --full --checkpoint-dir checkpoints --test-data data/generated/dataset.npz
python -m scripts.ablation --ablation all --data data/generated/dataset.npz
```

---

## Usage programmatique

```python
from packages.core.models import ExperimentGoal
from packages.orchestration.pipeline import CryoSwarmPipeline

goal = ExperimentGoal(
    title="Balayage robuste atomes neutres",
    scientific_objective="Chercher des protocoles de densité Rydberg robustes.",
    target_observable="rydberg_density",
    desired_atom_count=6,
    preferred_geometry="mixed",
)

pipeline = CryoSwarmPipeline(
    sequence_strategy_mode="adaptive",
    rl_checkpoint_path="checkpoints/ppo_latest.pt",
)

summary = pipeline.run(goal)
print(summary.status, summary.top_candidate_id)
```

Le pipeline retourne un `PipelineSummary` contenant l'état de la campagne, tous les candidats classés, les décisions agents, les rapports de robustesse et les enregistrements mémoire.

---

## Suite de tests

*372 tests. Pas un de trop, pas un de moins. Rutherford testait ses atomes un par un — nous on fait pareil mais plus vite.*

Le repo maintient **372 tests** couvrant :

| Domaine | Couverture |
|---------|------------|
| Construction du Hamiltonien (dense + sparse) | Exactitude physique, valeurs propres, symétrie |
| Observables quantiques | Densité Rydberg, corrélations, intrication |
| Agent géométrie | Contraintes d'espacement, conditions de blocage, bornes device |
| Agent séquence | 5 familles de formes d'onde, validation de paramètres |
| Profils de bruit | 3 scénarios, application des perturbations |
| Scoring de robustesse | Formule d'agrégation, bonus stabilité, pénalité |
| Scoring objectif | Score composite, normalisation des poids |
| Intégration pipeline | Exécution de campagne bout en bout |
| Pipeline parallèle | Évaluation de bruit concurrente |
| Endpoints API | Health check, CRUD campagnes, gestion d'erreurs |
| Modèle surrogate | Entraînement, inférence, incertitude d'ensemble |
| PPO + environnement RL | Espace d'action, observation, shaping de récompense |
| Dataset ML | Génération de features, normalisation |
| Routage backend | Logique de routage émulateur |
| Agent mémoire | Extraction de leçons, génération de tags |

Lancer avec couverture :

```bash
pytest tests -q --tb=short
```

---

## Cartographie du repo

*Le plan de la base. Chaque dossier a un rôle précis — pas de fourre-tout, pas de mystère. Leibniz voulait un langage universel pour les sciences — on a fait mieux : une arborescence bien rangée.*

```text
packages/                        Le cœur du système
├── core/                        Le socle — modèles de données, config, constantes
├── agents/                      8 agents spécialisés + switching de stratégie
├── orchestration/               Pipeline, phases, bus d'événements
├── simulation/                  Hamiltonien, observables, bruit, cache
├── scoring/                     Scoring de robustesse + objectif, classement
├── ml/                          Surrogate, PPO, env RL, apprentissage actif
├── pasqal_adapters/             Traducteurs vers Pulser, Pasqal Cloud, QoolQit
└── db/                          Persistance MongoDB

apps/
├── api/                         API REST FastAPI (5 routes + WebSocket)
└── dashboard/                   Dashboard Streamlit (9 pages)

scripts/                         CLI : entraînement, benchmarks, démo
configs/                         Hyperparamètres YAML
data/                            Datasets générés
checkpoints/                     Modèles entraînés (surrogate, PPO)
experiments/                     Tracking des runs d'entraînement
tests/                           372 tests
```

---

## Architecture détaillée — fichier par fichier

*Cette section, c'est le mode "capot ouvert". On t'explique chaque pièce du moteur, à quoi elle sert, et quel type de programme c'est. Comme disait Euler : "Rien n'est plus pratique qu'une bonne théorie" — et ici, rien n'est plus clair qu'une bonne cartographie.*

---

### `packages/core/` — Le socle commun

*Tout le monde importe depuis core. C'est les fondations de la maison — si c'est bancal ici, tout s'effondre.*

| Fichier | Type | Ce qu'il fait |
|---------|------|---------------|
| `models.py` | **Modèles de données Pydantic** | Définit TOUS les objets du système : `ExperimentGoal`, `ExperimentSpec`, `RegisterCandidate`, `SequenceCandidate`, `RobustnessReport`, `EvaluationResult`, `CampaignState`, `MemoryRecord`, `AgentDecision`... C'est le dictionnaire du projet — si tu veux comprendre une donnée, c'est ici. |
| `enums.py` | **Constantes énumérées** | Tous les statuts (`GoalStatus`, `CampaignStatus`), les backends (`BackendType`), les familles de séquences (`SequenceFamily`), les niveaux de bruit (`NoiseLevel`), les noms d'agents (`AgentName`). Ça évite les strings magiques partout. |
| `config.py` | **Configuration applicative** | Gère les settings avec Pydantic : URI MongoDB, clé API, credentials Pasqal, CORS, rate limiting. Détecte l'environnement (dev/test/prod). |
| `parameter_space.py` | **Espace des paramètres physiques** | Centralise TOUS les paramètres tunables : nombre d'atomes (2–50), espacement (4–15 μm), amplitude (1–15 rad/μs), désaccord (-40 à +25 rad/μs), coefficient $C_6$, profils de bruit. Permet l'échantillonnage Latin Hypercube pour générer des datasets. |
| `training_config.py` | **Config ML** | Charge les hyperparamètres depuis le YAML et dérive les échelles de normalisation des features à partir de l'espace physique. |
| `exceptions.py` | **Hiérarchie d'exceptions** | Exceptions typées par domaine : `AgentError`, `GeometryError`, `SequenceError`, `PipelineError`... Chaque erreur porte le nom de l'agent responsable. |
| `metadata_schemas.py` | **Contrats de données (TypedDict)** | Schémas typés pour les métadonnées échangées entre agents : `RegisterMetadata`, `SequenceMetadata`, `EvaluationMetadata`, `MemorySignals`. |
| `logging.py` | **Logging** | Configuration du logger singleton. On sait tout ce qui se passe, quand, et pourquoi. |

---

### `packages/agents/` — Les 8 agents spécialisés

*Chaque agent fait UN truc et le fait bien. C'est le principe de responsabilité unique — Uncle Bob serait content, et Turing aussi d'ailleurs.*

| Fichier | Type | Ce qu'il fait |
|---------|------|---------------|
| `base.py` | **Classe abstraite** | `BaseAgent` — la classe mère de tous les agents. Fournit le logger et la méthode `build_decision()` qui produit des `AgentDecision` traçables (horodatées, avec raisonnement). |
| `protocols.py` | **Interfaces (Protocol)** | Contrats runtime-checkable : `ProblemFramingProtocol`, `GeometryProtocol`, `SequenceProtocol`, etc. C'est ce qui garantit que chaque agent respecte son API sans héritage rigide. |
| `problem_agent.py` | **Agent de cadrage** | `ProblemFramingAgent` — prend un `ExperimentGoal` ("je veux étudier la densité Rydberg sur 6 atomes") et le transforme en `ExperimentSpec` (nombre d'atomes exact, géométrie, familles de séquences à explorer, densité cible). Utilise la mémoire pour biaiser le choix de backend. |
| `geometry_agent.py` | **Agent de géométrie** | `GeometryAgent` — génère des registres atomiques faisables hardware. Sait créer des grilles carrées, des lignes, des triangulaires, des anneaux, des zigzags, des nids d'abeille. Vérifie les contraintes : espacement min, rayon de blocage, bornes du device. Calcule la matrice de Van der Waals. |
| `sequence_agent.py` | **Agent de séquences** | `SequenceAgent` — génère des candidats pulse par heuristiques. Balaye les 5 familles × variations de paramètres. Prédit le coût d'exécution. C'est la version "force brute intelligente" de la génération. |
| `noise_agent.py` | **Agent de robustesse bruit** | `NoiseRobustnessAgent` — évalue chaque candidat sous bruit. Lance la simulation nominale + 3 scénarios perturbés (BAS/MOYEN/STRESSÉ). Produit un `RobustnessReport` avec scores nominal, moyen, pire cas. |
| `routing_agent.py` | **Agent de routage** | `BackendRoutingAgent` — décide quel simulateur utiliser. Petits systèmes (≤8 atomes) → `EMU_SV` (state vector exact). Moyens (≤16 atomes) → `EMU_MPS` (tensor network approché). Sinon → `LOCAL_PULSER`. |
| `campaign_agent.py` | **Agent de campagne** | `CampaignAgent` — classe tous les candidats évalués par score composite (lexicographique : objectif > pire cas > robustesse > nominal). Sélectionne le meilleur, met à jour le statut de la campagne. |
| `memory_agent.py` | **Agent de mémoire** | `MemoryAgent` — après chaque campagne, extrait les top 3 candidats et crée des `MemoryRecord` taggés. Stocke les signaux : scores, backend, famille, layout, confiance, gap spectral. Ces leçons alimentent les futures campagnes. |
| `results_agent.py` | **Agent de résultats** | `ResultsAgent` — génère le rapport final : résumé de l'objectif, de la spec, du meilleur candidat, du backend recommandé, des scores. |
| `sequence_strategy.py` | **Sélecteur de stratégie** | `SequenceStrategy` — décide COMMENT générer les séquences. 5 modes : `HEURISTIC_ONLY`, `RL_ONLY`, `HYBRID`, `ADAPTIVE`, `BANDIT`. Le mode bandit utilise UCB1 pour explorer/exploiter les stratégies automatiquement. |

---

### `packages/orchestration/` — Le chef d'orchestre

*C'est lui qui dit "toi tu fais ça, toi tu passes après, toi tu notes". Sans orchestration, les agents seraient des musiciens qui jouent chacun dans leur coin.*

| Fichier | Type | Ce qu'il fait |
|---------|------|---------------|
| `pipeline.py` | **Pipeline principal** | `CryoSwarmPipeline` — LE point d'entrée. Crée le contexte, enchaîne les phases dans l'ordre, gère la parallélisation de l'évaluation bruit, propage le contexte entre les phases. C'est la méthode `run()` qui lance tout. |
| `phases.py` | **Phases composables** | 9 classes de phase : `ProblemFramingPhase`, `GeometryGenerationPhase`, `SequenceGenerationPhase`, `SurrogateFilterPhase`, `EvaluationPhase`, `RankingPhase`, `MemoryCapturePhase`, `ResultsSummaryPhase`. Chaque phase opère sur un `PipelineContext` mutable et publie des événements. |
| `events.py` | **Bus d'événements** | `EventBus` — système pub/sub in-process. Chaque phase publie des `PipelineEvent` (début, fin, erreur). Le dashboard s'y abonne pour le monitoring en temps réel. Supporte le replay et les abonnements par campagne. |
| `runner.py` | **Lanceur de démo** | `run_demo_campaign()` — raccourci pour lancer une campagne démo complète avec un objectif prédéfini. |

---

### `packages/simulation/` — Là où la physique se calcule

*C'est le labo. Ici on diagonalise des Hamiltoniens, on calcule des observables, on simule du bruit. Schrödinger ferait tourner ce code sur son chat (s'il en avait un).*

| Fichier | Type | Ce qu'il fait |
|---------|------|---------------|
| `hamiltonian.py` | **Construction du Hamiltonien** | Construit la matrice Hamiltonienne Rydberg dense ($2^N \times 2^N$) ou sparse. Calcule les distances inter-atomiques, la matrice de Van der Waals ($C_6/r^6$), le rayon de blocage ($R_b = (C_6/\Omega)^{1/6}$), et résout le MIS (Maximum Independent Set) par énumération exacte ($N \leq 15$) ou heuristique gloutone. Convention MSB (Most Significant Bit). |
| `observables.py` | **Calcul des observables** | À partir du vecteur d'état $|\psi\rangle$, calcule : densité Rydberg par site $\langle \hat{n}_i \rangle$, fraction totale, corrélation paire, corrélation connectée $g^{(2)}_{ij}$, ordre antiferromagnétique, entropie d'intrication (via SVD de la matrice densité réduite), probabilités de bitstring, fidélité d'état. |
| `evaluators.py` | **Moteur d'évaluation** | `evaluate_candidate_robustness()` — orchestre l'évaluation complète d'un candidat. Lance la simulation Pulser QutipEmulator (ou le backend numpy) en nominal + perturbé. Calcule les observables et les métriques Hamiltoniennes. Met en cache les résultats. |
| `noise_profiles.py` | **Profils de bruit** | Trois scénarios prédéfinis : `low_noise()` ($\sigma_{amp}=0.01$), `medium_noise()` ($\sigma_{amp}=0.05$), `stressed_noise()` ($\sigma_{amp}=0.10$). Chaque profil inclut bruit d'amplitude, dérive du désaccord, déphasage, perte d'atomes, SPAM, inhomogénéité spatiale. |
| `numpy_backend.py` | **Simulateur exact NumPy** | Backend de simulation indépendant de Pulser. Implémente l'évolution temporelle par splitting de Strang (Trotter-Suzuki ordre 2). Définit un `PulseSchedule` pour décrire les formes d'onde. Fallback quand Pulser n'est pas installé. |
| `evaluation_cache.py` | **Cache de simulation** | Cache LRU adressable par contenu. Hash la spec, le registre, la séquence et le scénario de bruit en une signature sémantique. Évite de re-simuler ce qu'on a déjà calculé. |

---

### `packages/scoring/` — Le jury

*3 fichiers, 3 responsabilités. Pas de bavardage — ici on note et on classe.*

| Fichier | Type | Ce qu'il fait |
|---------|------|---------------|
| `robustness.py` | **Score de robustesse** | Calcule l'agrégat : $S_{robust} = 0.25 \cdot s_{nom} + 0.35 \cdot \bar{s}_{pert} + 0.30 \cdot s_{worst} + 0.10 \cdot b_{stab}$. Bonus de stabilité si la variance est faible. Pénalité si le pire cas dégringole par rapport au nominal. |
| `objective.py` | **Score objectif composite** | $S_{obj} = 0.45 \cdot s_{obs} + 0.35 \cdot S_{robust} - 0.10 \cdot c_{exec} - 0.10 \cdot \ell_{latence}$. C'est LE nombre final qui détermine si un candidat est bon. |
| `ranking.py` | **Classement** | Tri lexicographique : score objectif > pire cas > robustesse > nominal. Assigne un rang final et le statut `RANKED` à chaque candidat. |

---

### `packages/ml/` — Le labo Machine Learning

*C'est ici que vit le PPO, le surrogate, l'apprentissage actif. Pour ceux qui cherchent "mais il est où le réseau de neurones ?" — c'est là.*

#### Surrogate (le réseau qui prédit sans simuler)

| Fichier | Type | Ce qu'il fait |
|---------|------|---------------|
| `surrogate.py` | **Réseau de neurones PyTorch** | `SurrogateModel` — MLP avec 3 blocs résiduels (LayerNorm → Linear → GELU → Dropout → Linear). Entrée : 18 features physiques. Sortie : 4 scores (robustesse, nominal, pire cas, observable). `SurrogateEnsemble` — 3 modèles indépendants dont on utilise la variance comme mesure d'incertitude. |
| `dataset.py` | **Préparation des données** | `build_feature_vector_v2()` — transforme un registre + séquence en vecteur 18-dim (nombre d'atomes, ratios d'espacement, métriques de blocage, paramètres de pulse, encodage layout/famille). `MLDataset` — PyTorch Dataset standard. |
| `data_generator.py` | **Génération de datasets** | `DataGenerator` — génère des milliers de configurations par échantillonnage Latin Hypercube / Sobol, les simule, et stocke les résultats en `.npz`. C'est ce qui produit les données d'entraînement pour le surrogate. |
| `normalizer.py` | **Normalisation** | `DatasetNormalizer` — fit mean/std sur les données d'entraînement, transform/inverse_transform, sérialise en .npz. |
| `surrogate_filter.py` | **Filtre pré-simulation** | `SurrogateFilter` — utilise l'ensemble surrogate pour pré-trier les candidats AVANT la simulation complète. Garde les top-k par score prédit, exclut ceux avec trop d'incertitude. |

#### PPO et Reinforcement Learning (l'agent qui apprend à designer des pulses)

| Fichier | Type | Ce qu'il fait |
|---------|------|---------------|
| `ppo.py` | **🎯 L'algo PPO (Proximal Policy Optimization)** | C'est ICI que vit le PPO. `ActorCritic` — réseau PyTorch avec deux têtes : l'**acteur** (propose des actions = paramètres de pulse) et le **critique** (estime la valeur de l'état = "est-ce que cette situation est bonne ?"). `PPOTrainer` — implémente l'algorithme PPO : collecte des rollouts, calcule l'avantage GAE (Generalized Advantage Estimation), optimise avec le loss clippé $L^{CLIP}$, met à jour le réseau. Learning rate scheduler avec warmup. |
| `rl_env.py` | **Environnement RL (style Gymnasium)** | `PulseDesignEnv` — l'environnement dans lequel le PPO apprend. **Observation** (16-dim) : nombre d'atomes, espacement, rayon de blocage, faisabilité, densité cible, meilleure robustesse vue, meilleurs paramètres, progression. **Action** (4-dim continu) : amplitude Ω, désaccord δ, durée T, choix de famille de pulse. **Récompense** : le score objectif du candidat généré. C'est la boucle observation → action → récompense → apprentissage. |
| `rl_sequence_agent.py` | **Agent RL pour les séquences** | `RLSequenceAgent` — wrapper qui prend la politique PPO entraînée et l'utilise comme générateur de séquences dans le pipeline. Si le PPO n'est pas entraîné → fallback automatique sur l'heuristique. |
| `curriculum.py` | **Curriculum learning** | `CurriculumScheduler` — entraîne le PPO progressivement : d'abord sur des petits systèmes (3 atomes), puis moyens (8), puis grands (16). Comme un prof qui commence par les exos faciles. Seuils de performance minimaux pour passer au niveau suivant. |

#### Apprentissage actif (le cercle vertueux)

| Fichier | Type | Ce qu'il fait |
|---------|------|---------------|
| `active_learning.py` | **Boucle d'apprentissage actif** | `ActiveLearningLoop` — le cycle complet : (1) entraîner le surrogate, (2) entraîner le PPO avec le surrogate comme proxy de récompense, (3) identifier les points à forte incertitude (là où le surrogate doute), (4) simuler ces points pour de vrai, (5) enrichir le dataset, (6) recommencer. C'est comme un étudiant qui révise uniquement ce qu'il ne maîtrise pas. |

#### Tracking et GPU

| Fichier | Type | Ce qu'il fait |
|---------|------|---------------|
| `experiment_tracking.py` | **Tracking CSV** | `ExperimentTracker` — log chaque run d'entraînement : commit git, device, dataset, métriques par step (loss, reward), registre d'artefacts. |
| `training_runner.py` | **Orchestrateur d'entraînement** | `train_surrogate()`, `train_rl()`, `train_active_learning()` — charge les données, gère les checkpoints, lance le curriculum. |
| `gpu_backend.py` | **Accélération GPU** | Construction du Hamiltonien sparse sur GPU (PyTorch). Approximation de Krylov pour l'évolution temporelle. Détection automatique : CUDA / ROCm (AMD MI300X) / MPS (Apple) / CPU. |

---

### `packages/pasqal_adapters/` — Les traducteurs vers le hardware

*CryoSwarm-Q parle son propre langage en interne. Ces adaptateurs traduisent vers l'écosystème Pasqal — comme des interprètes entre deux pays.*

| Fichier | Type | Ce qu'il fait |
|---------|------|---------------|
| `pulser_adapter.py` | **Adaptateur Pulser** | Convertit les `RegisterCandidate` et `SequenceCandidate` internes en objets Pulser natifs (`Register`, `Sequence`). Gère les 5 familles de formes d'onde (Ramp, Blackman, Constant), clippe les amplitudes pour marge de sécurité, quantise la durée sur la période d'horloge. |
| `emulator_router.py` | **Routeur d'émulateur** | `recommend_backend()` — décide quel simulateur utiliser en fonction de la taille du système et du profil de robustesse. ≤8 atomes → state vector exact (EMU_SV). ≤16 atomes → tensor network approché (EMU_MPS). Sinon → Pulser local. |
| `pasqal_cloud_adapter.py` | **Adaptateur Pasqal Cloud** | Interface vers le SDK cloud Pasqal. Authentification, soumission de batches, vérification de statut. Dégradation gracieuse si les credentials sont absents. |
| `qoolqit_adapter.py` | **Bridge QoolQit** | Encode le graphe d'interaction de blocage en QUBO (Quadratic Unconstrained Binary Optimization) pour résolution MIS via QoolQit. |

---

### `packages/db/` — La mémoire persistante

| Fichier | Type | Ce qu'il fait |
|---------|------|---------------|
| `mongodb.py` | **Client MongoDB** | Connexion singleton thread-safe avec pool management, timeouts, config de taille de pool. |
| `repositories.py` | **Couche Repository** | `CryoSwarmRepository` — pattern Repository avec CRUD complet pour tous les modèles : `create_goal()`, `get_campaign()`, `list_candidates_for_campaign()`, `store_memory_record()`, etc. |
| `init_db.py` | **Initialisation** | Crée les collections et les index (unique sur `id`, index sur `campaign_id`, `goal_id`, timestamps) au premier démarrage. |

---

### `apps/api/` — L'API REST FastAPI

*Pour ceux qui veulent interagir avec CryoSwarm-Q sans le dashboard — ou qui construisent un frontend custom.*

| Fichier | Type | Ce qu'il fait |
|---------|------|---------------|
| `main.py` | **Application FastAPI** | Factory : lifespan (init MongoDB au démarrage, close au shutdown), middleware CORS, rate limiting, broadcast d'événements. |
| `auth.py` | **Authentification** | Vérification API key (constant-time) via header `X-API-Key`. Optionnel si pas de clé configurée. |
| `dependencies.py` | **Injection de dépendances** | Fournit le `CryoSwarmRepository` injecté dans chaque route qui en a besoin. |
| `rate_limit.py` | **Rate limiting** | Limiteur de requêtes par fenêtre fixe. Track par client, retourne le quota restant et le temps de retry. |
| `live.py` | **Broadcast temps réel** | `CampaignEventBroadcaster` — pousse les événements pipeline vers les clients WebSocket par `campaign_id`. |
| `routes/health.py` | **GET /health** | Statut de l'app, de l'environnement, de la connectivité MongoDB. |
| `routes/goals.py` | **POST/GET /goals** | Création et récupération d'objectifs expérimentaux. |
| `routes/campaigns.py` | **POST/GET /campaigns** | Lancement de campagnes démo, récupération de l'état complet d'une campagne. |
| `routes/candidates.py` | **GET /candidates** | Liste des candidats évalués et classés pour une campagne donnée. |
| `routes/streaming.py` | **WebSocket /ws/campaigns** | Streaming live des événements pipeline vers le navigateur. |

---

### `apps/dashboard/` — Le dashboard Streamlit (9 pages)

| Fichier | Ce qu'il fait |
|---------|---------------|
| `app.py` | Page d'accueil : sidebar avec liste des campagnes, métriques globales, liens vers les sous-pages. |
| `logic.py` | Helpers Python purs (testables sans Streamlit) : formatage des tables, calcul de Pareto, résumé d'événements. |
| `components/data_loaders.py` | Chargement des données depuis MongoDB ou le repo. |
| `components/latex_panels.py` | 7 générateurs de panneaux LaTeX : formules du Hamiltonien, observables, MIS, pulses, robustesse, ML, campagne. |
| `components/plotly_charts.py` | 20+ fonctions de visualisation Plotly : spectres, heatmaps, barres, radar, violins, 3D. |
| `pages/1_Campaign_Control.py` | Lancer et monitorer des campagnes en temps réel. |
| `pages/2_Register_Physics.py` | Explorer la physique des registres : scatter 2D, heatmap VdW, graphe de blocage. |
| `pages/3_Hamiltonian_Lab.py` | Spectroscopie : diagonalisation, spectre d'énergie, gap spectral, IPR. |
| `pages/4_Pulse_Studio.py` | Design de séquences : visualisation des formes d'onde, comparaison, générateur. |
| `pages/5_Robustness_Arena.py` | Analyse de robustesse : barres groupées, radar, violin, cascade de dégradation. |
| `pages/6_ML_Observatory.py` | Monitoring ML : loss surrogate, reward PPO, bandit UCB1, RL vs heuristique. |
| `pages/7_Campaign_Analytics.py` | Tendances inter-campagnes, front de Pareto 3D, nuage de tags mémoire. |
| `pages/8_Theory_Reference.py` | Référence mathématique complète (300+ lignes) : 8 sections dépliables. |
| `pages/9_Training_Tracker.py` | Suivi des runs d'entraînement : métriques, courbes, observations. |

---

### `scripts/` — Les outils en ligne de commande

| Fichier | Ce qu'il fait |
|---------|---------------|
| `train_ml.py` | **CLI d'entraînement ML** — 6 phases : `generate` (créer des données), `generate_v2` (dataset LHS grande échelle), `surrogate` (entraîner le surrogate ~200 epochs), `rl` (PPO ~500 updates), `active` (boucle d'apprentissage actif), `full` (surrogate + PPO bout en bout). |
| `benchmark.py` | **Suite de benchmarks** — MSE/MAE/R² pour le surrogate, récompense/convergence pour le PPO, robustesse moyenne pour le pipeline. Inclut hash git et timestamps. |
| `ablation.py` | **Études d'ablation** — 9 configurations testées : heuristique pure, surrogate filter v1/v2, RL single/multi-step, curriculum, hybride, ensemble 3 modèles, pipeline complet. Compare pour voir ce qui marche vraiment. |
| `run_demo_pipeline.py` | **Démo rapide** — lance une campagne complète et affiche le meilleur candidat. |
| `seed_demo_goal.py` | **Seed d'objectif** — crée un objectif test en base pour les tests de la chaîne. |

---

### Vue d'ensemble architecturale

```text
┌──────────────────────────────────────────────────────────┐
│  DASHBOARD Streamlit (9 pages)                           │
│  → L'interface visuelle pour tout inspecter              │
└──────────────────────┬───────────────────────────────────┘
                       │ HTTP / WebSocket
┌──────────────────────▼───────────────────────────────────┐
│  API FastAPI (5 routes + WebSocket live)                  │
│  → Le point d'entrée programmatique                      │
└──────────────────────┬───────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────┐
│  ORCHESTRATION (Pipeline + 9 Phases + EventBus)          │
│  → Le chef d'orchestre qui enchaîne les agents           │
│                                                          │
│  Cadrage → Géométrie → Séquences → Filtre surrogate     │
│  → Évaluation bruit → Classement → Mémoire → Résultats  │
└───┬──────────────┬──────────────┬────────────────────────┘
    │              │              │
    ▼              ▼              ▼
┌────────┐  ┌──────────┐  ┌───────────────┐
│SIMULA- │  │ SCORING  │  │ ML PIPELINE   │
│TION    │  │          │  │               │
│        │  │objective │  │surrogate.py   │
│hamilto-│  │robustness│  │  → 3 réseaux  │
│nian.py │  │ranking   │  │ppo.py         │
│observa-│  │          │  │  → PPO actor- │
│bles.py │  │          │  │    critic     │
│noise   │  │          │  │rl_env.py      │
│evaluat-│  │          │  │  → environnmt │
│ors     │  │          │  │active_learn.  │
│cache   │  │          │  │  → boucle AL  │
└───┬────┘  └──────────┘  └───────┬───────┘
    │                             │
    ▼                             ▼
┌──────────────────────────────────────────┐
│  PASQAL ADAPTERS                         │
│  pulser_adapter → Pulser natif           │
│  emulator_router → EMU_SV / EMU_MPS     │
│  pasqal_cloud → SDK Cloud                │
│  qoolqit_adapter → QUBO / MIS           │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  PERSISTENCE : MongoDB (packages/db/)    │
│  → Campagnes, candidats, mémoire, runs   │
└──────────────────────────────────────────┘
```

---

## Limitations actuelles

*L'honnêteté intellectuelle, c'est pas optionnel. Voici ce qu'on ne prétend pas faire (pour l'instant).*

- Prototype de recherche — pas calibré contre un device physique spécifique.
- Diagonalisation dense du Hamiltonien limitée à $N \leq 14$ atomes (les méthodes sparse étendent cette plage).
- Les modèles surrogate et PPO nécessitent un entraînement avant utilisation ; sans entraînement, le système tombe sur les stratégies heuristiques.
- Les campagnes d'apprentissage actif à grande échelle demandent du compute sérieux (GPU recommandé).
- Les décisions du pipeline sont principées mais pas encore validées contre des données expérimentales.

---

## Licence

Prototype de recherche. Consulter le contenu du repository pour les termes du projet.
