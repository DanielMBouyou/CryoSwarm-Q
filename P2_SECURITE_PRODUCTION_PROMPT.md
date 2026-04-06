# P2 — SÉCURITÉ & PRODUCTION : Prompt d'implémentation exhaustif

> **Destinataire** : Agent IA de code.
> **Mode** : Implémentation directe — écris chaque ligne, chaque import, chaque test.
> **Philosophie** : Sécurité production-grade sans over-engineering. Chaque modification doit être motivée par un risque concret (OWASP Top 10, CWE, PyTorch security advisory). C'est un projet de quantum computing en recherche — la posture de sécurité doit être sérieuse sans transformer le prototype en forteresse inutile.

---

## Contexte du projet

CryoSwarm-Q est un système multi-agent hardware-aware pour la conception autonome d'expériences en informatique quantique à atomes neutres. L'API FastAPI et le dashboard Streamlit sont les deux points d'entrée exposés. MongoDB Atlas est le backend de persistance.

**Fichier manifeste** : `CLAUDE.md` à la racine contient la vision complète du projet.

**Hypothèse** : Les tâches P0 et P1 ont été implémentées. Si certaines ne sont pas encore en place, adapte-toi — ne casse rien.

**Règles impératives** :
- Python 3.11+, types partout, `from __future__ import annotations`
- Pydantic v2.8+
- Imports absolus uniquement (`from packages.core.models import ...`)
- Chaque import groupé : stdlib, third-party, project
- `pytest tests/ -x --tb=short` après CHAQUE tâche
- Zéro `# TODO` laissé dans le code produit
- Zéro credential en dur, zéro secret exposé

---

## TÂCHE 1 — Masquer les détails d'erreur dans les réponses API

### Problème actuel

**Fichier** : `apps/api/routes/campaigns.py` (lignes 23-29)

Le handler `run_demo()` retourne `str(exc)` dans le corps JSON de la réponse 500 :

```python
raise HTTPException(
    status_code=500,
    detail={"error": "Campaign execution failed.", "message": str(exc)},
) from exc
```

Cela viole **OWASP A05:2021 — Security Misconfiguration** et **CWE-209 (Information Exposure Through an Error Message)**. Un attaquant peut lire les tracebacks internes, les noms de modules, les chemins de fichiers, et les configurations MongoDB.

**Fichier** : `apps/api/routes/health.py` (lignes 27-28)

Le health endpoint expose les détails d'exception MongoDB :

```python
except Exception as exc:
    payload["mongodb_ping"] = f"failed: {exc}"
```

Cela peut révéler l'URI MongoDB Atlas, le nom d'hôte du cluster, les timeouts — informations utiles pour un attaquant.

### Objectif

- Les réponses d'erreur en production NE doivent JAMAIS contenir de traceback, nom de fichier, message d'exception interne, ou détail d'infrastructure.
- En mode développement (`APP_ENV=development`), les détails peuvent être retournés pour faciliter le debug.
- Les détails complets sont TOUJOURS loggés côté serveur.

### Modifications

#### 1. `apps/api/routes/campaigns.py`

```python
# Avant :
raise HTTPException(
    status_code=500,
    detail={"error": "Campaign execution failed.", "message": str(exc)},
) from exc

# Après :
from packages.core.config import get_settings
from packages.core.enums import AppEnvironment

# ...dans run_demo() :
except Exception as exc:
    logger.error("Demo campaign failed: %s", exc, exc_info=True)
    settings = get_settings()
    detail: dict[str, str] = {"error": "Campaign execution failed."}
    if settings.app_env == AppEnvironment.DEVELOPMENT:
        detail["debug_message"] = str(exc)
    raise HTTPException(status_code=500, detail=detail) from exc
```

#### 2. `apps/api/routes/health.py`

```python
# Avant :
except Exception as exc:
    payload["mongodb_ping"] = f"failed: {exc}"

# Après :
except Exception as exc:
    logger.warning("MongoDB ping failed: %s", exc)
    payload["mongodb_ping"] = "failed"
```

Ajouter l'import du logger en haut du fichier :
```python
from packages.core.logging import get_logger

logger = get_logger(__name__)
```

#### 3. `apps/api/main.py` — Global exception handler

Le handler global est déjà correct (il ne retourne pas `str(exc)`). Il doit rester tel quel. Vérifier simplement qu'il logge le traceback :

```python
@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "Unhandled error on %s %s: %s",
        request.method,
        request.url.path,
        exc,
        exc_info=True,     # <-- AJOUTER : loguer le traceback complet
    )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error."},
    )
```

**Retirer** aussi le texte « Check server logs for details. » du message — il indique aux attaquants que les logs sont un vecteur intéressant.

### Tests

Modifier `tests/test_api_error_handling.py` :

```python
def test_run_demo_returns_structured_500_when_pipeline_crashes(monkeypatch) -> None:
    app.dependency_overrides[get_repository] = lambda: _FakeRepository()
    client = TestClient(app)
    monkeypatch.setattr(
        campaigns_route_module,
        "run_demo_campaign",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    response = client.post("/campaigns/run-demo", json={})

    assert response.status_code == 500
    body = response.json()
    assert body["detail"]["error"] == "Campaign execution failed."
    # En mode development, le debug_message est présent
    # Mais JAMAIS un traceback complet
    if "debug_message" in body["detail"]:
        assert "boom" in body["detail"]["debug_message"]
    app.dependency_overrides.clear()


def test_global_handler_hides_internal_exception_details() -> None:
    def _raise_boom() -> None:
        raise RuntimeError("sensitive internal path /etc/mongodb.conf leaked")

    app.add_api_route("/_test/unhandled", _raise_boom, methods=["GET"])
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/_test/unhandled")

    assert response.status_code == 500
    body = response.json()
    assert body == {"error": "Internal server error."}
    # S'assurer qu'aucun détail ne fuite
    assert "sensitive" not in str(body)
    assert "mongodb" not in str(body).lower()
    app.router.routes.pop()
```

Ajouter un test pour le health endpoint :

```python
def test_health_does_not_leak_mongodb_uri(monkeypatch) -> None:
    """Ensures the health endpoint does not expose MongoDB connection details."""
    import packages.db.mongodb as mongodb_module

    original_get_client = mongodb_module.get_mongo_client

    def _failing_client(*args, **kwargs):
        raise ConnectionError("mongodb+srv://admin:s3cret@cluster0.example.net/db?retryWrites=true")

    monkeypatch.setattr(mongodb_module, "get_mongo_client", _failing_client)
    # Force has_mongodb = True for this test
    from packages.core.config import Settings, get_settings
    monkeypatch.setattr(
        "apps.api.routes.health.get_settings",
        lambda: Settings(mongodb_uri="mongodb://fake", app_env="development"),
    )

    client = TestClient(app)
    response = client.get("/health")

    body = response.json()
    assert "s3cret" not in str(body)
    assert "admin" not in str(body)
    assert body.get("mongodb_ping") == "failed"
```

---

## TÂCHE 2 — Ajouter authentification minimale (API key)

### Choix d'architecture

Pour un prototype de recherche, le **JWT est excessif** — il nécessite un système de login, un token store, un refresh flow. Une **API key statique via header** est le bon compromis : simple, sécurisée, suffisante pour protéger le prototype contre les appels non autorisés.

La clé est lue depuis la variable d'environnement `CRYOSWARM_API_KEY`. Si la variable n'est pas configurée (développement local), l'authentification est désactivée.

### Modifications

#### 1. Ajouter le champ dans `packages/core/config.py`

```python
class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # ... champs existants ...
    api_key: str = ""                    # <-- NOUVEAU

    @property
    def has_api_key(self) -> bool:       # <-- NOUVEAU
        return bool(self.api_key)
```

Et dans `get_settings()` :

```python
api_key=os.getenv("CRYOSWARM_API_KEY", ""),
```

#### 2. Créer `apps/api/auth.py`

```python
"""API key authentication for CryoSwarm-Q.

When CRYOSWARM_API_KEY is set, all mutating endpoints require
the key in the X-API-Key header. When the key is empty (local
development), authentication is bypassed.
"""
from __future__ import annotations

import secrets

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

from packages.core.config import Settings, get_settings


_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(
    api_key: str | None = Security(_API_KEY_HEADER),
    settings: Settings = Depends(get_settings),
) -> None:
    """Verify the API key from the X-API-Key header.

    Skips verification when CRYOSWARM_API_KEY is not configured,
    allowing local development without authentication.
    """
    if not settings.has_api_key:
        return

    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide X-API-Key header.",
        )

    # Constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(api_key, settings.api_key):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key.",
        )
```

**Points de sécurité** :
- `secrets.compare_digest()` empêche les timing attacks (CWE-208)
- `auto_error=False` permet de retourner un message custom au lieu du 403 par défaut
- La dépendance est injectable et testable

#### 3. Appliquer l'authentification aux routes mutantes

L'authentification est requise sur :
- `POST /goals` (création de données)
- `POST /campaigns/run-demo` (déclenche un pipeline lourd)

L'authentification n'est PAS requise sur :
- `GET /health` (monitoring, doit être accessible sans clé)
- `GET /goals/{id}` (lecture seule, acceptable en prototype)
- `GET /campaigns/{id}` (lecture seule)
- `GET /campaigns/{id}/candidates` (lecture seule)

```python
# apps/api/routes/goals.py
from apps.api.auth import verify_api_key

@router.post("", response_model=ExperimentGoal, dependencies=[Depends(verify_api_key)])
def create_goal(
    payload: ExperimentGoalCreate,
    repository: CryoSwarmRepository = Depends(get_repository),
) -> ExperimentGoal:
    ...
```

```python
# apps/api/routes/campaigns.py
from apps.api.auth import verify_api_key

@router.post("/run-demo", response_model=PipelineSummary, dependencies=[Depends(verify_api_key)])
def run_demo(
    payload: DemoGoalRequest | None = None,
    repository: CryoSwarmRepository = Depends(get_repository),
) -> PipelineSummary:
    ...
```

#### 4. Tests

Créer `tests/test_api_auth.py` :

```python
"""API key authentication tests."""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from apps.api.dependencies import get_repository
from apps.api.main import app
from packages.core.config import Settings


class _FakeRepository:
    def create_goal(self, goal):
        return goal

    def get_goal(self, goal_id: str):
        return None

    def get_campaign(self, campaign_id: str):
        return None

    def list_candidates_for_campaign(self, campaign_id: str):
        return []


@pytest.fixture()
def _fake_repo():
    app.dependency_overrides[get_repository] = lambda: _FakeRepository()
    yield
    app.dependency_overrides.clear()


def _settings_with_key(key: str = "test-secret-key-12345") -> Settings:
    """Create settings with an API key configured."""
    return Settings(
        api_key=key,
        mongodb_uri="mongodb://fake",
    )


def _settings_without_key() -> Settings:
    """Create settings without an API key (dev mode)."""
    return Settings(mongodb_uri="mongodb://fake")


class TestApiKeyRequired:
    """Test behavior when CRYOSWARM_API_KEY is set."""

    def test_post_goal_without_key_returns_401(self, _fake_repo, monkeypatch) -> None:
        monkeypatch.setattr("packages.core.config.get_settings", _settings_with_key)
        client = TestClient(app)

        response = client.post(
            "/goals",
            json={
                "title": "Test goal auth",
                "scientific_objective": "Test auth enforcement.",
            },
        )

        assert response.status_code == 401
        assert "Missing API key" in response.json()["detail"]

    def test_post_goal_with_wrong_key_returns_403(self, _fake_repo, monkeypatch) -> None:
        monkeypatch.setattr("packages.core.config.get_settings", _settings_with_key)
        client = TestClient(app)

        response = client.post(
            "/goals",
            json={
                "title": "Test goal auth",
                "scientific_objective": "Test auth enforcement.",
            },
            headers={"X-API-Key": "wrong-key"},
        )

        assert response.status_code == 403
        assert "Invalid API key" in response.json()["detail"]

    def test_post_goal_with_correct_key_succeeds(self, _fake_repo, monkeypatch) -> None:
        monkeypatch.setattr("packages.core.config.get_settings", _settings_with_key)
        client = TestClient(app)

        response = client.post(
            "/goals",
            json={
                "title": "Test goal auth success",
                "scientific_objective": "Test that valid key passes.",
            },
            headers={"X-API-Key": "test-secret-key-12345"},
        )

        assert response.status_code == 200

    def test_get_health_without_key_succeeds(self, _fake_repo, monkeypatch) -> None:
        """Health endpoint must remain accessible without auth."""
        monkeypatch.setattr("packages.core.config.get_settings", _settings_with_key)
        client = TestClient(app)

        response = client.get("/health")

        assert response.status_code == 200

    def test_get_goal_without_key_succeeds(self, _fake_repo, monkeypatch) -> None:
        """GET endpoints are accessible without auth."""
        monkeypatch.setattr("packages.core.config.get_settings", _settings_with_key)
        client = TestClient(app)

        response = client.get("/goals/any-id")

        # 404 is expected (fake repo), NOT 401/403
        assert response.status_code == 404


class TestApiKeyDisabled:
    """Test behavior when CRYOSWARM_API_KEY is not set (local dev)."""

    def test_post_goal_without_key_succeeds_in_dev(self, _fake_repo, monkeypatch) -> None:
        monkeypatch.setattr("packages.core.config.get_settings", _settings_without_key)
        client = TestClient(app)

        response = client.post(
            "/goals",
            json={
                "title": "Test goal no auth",
                "scientific_objective": "Auth disabled in dev.",
            },
        )

        assert response.status_code == 200


class TestTimingSafety:
    """Verify constant-time comparison is used."""

    def test_compare_digest_is_used(self) -> None:
        """Verify the auth module uses secrets.compare_digest."""
        import inspect
        from apps.api import auth
        source = inspect.getsource(auth.verify_api_key)
        assert "compare_digest" in source
```

---

## TÂCHE 3 — Utiliser `weights_only=True` dans `torch.load()`

### Problème actuel

6 appels à `torch.load(..., weights_only=False)` dans le codebase :

| Fichier | Ligne | Contexte |
|---------|-------|----------|
| `packages/ml/surrogate_filter.py` | 70 | `_load_model()` — charge un checkpoint pour le filtre surrogate |
| `packages/ml/ppo.py` | 98 | `from_checkpoint()` — reconstruit un ActorCritic depuis un .pt |
| `packages/ml/ppo.py` | 161 | `load()` — charge state_dict dans un modèle existant |
| `packages/ml/surrogate.py` | 109 | SurrogateModel `load()` |
| `packages/ml/surrogate.py` | 210 | SurrogateModelV2 `load()` |
| `scripts/benchmark.py` | 153 | charge un modèle pour benchmark |

`weights_only=False` désactive la protection contre la **désérialisation d'objets Python arbitraires** (CWE-502 — Deserialization of Untrusted Data). Un fichier `.pt` malveillant peut exécuter du code arbitraire à l'ouverture.

### Analyse de faisabilité

Les checkpoints sauvés par le projet contiennent :
- `state_dict()` → tensors PyTorch (safe avec `weights_only=True`)
- `normalizer_means`, `normalizer_stds` → `list[float] | None` (safe)
- `config` → `dict[str, int | float | str]` (safe)
- `version` → `str` (safe)

Tous ces types sont dans la whitelist PyTorch pour `weights_only=True`. La migration est donc **directe et sans risque** pour les checkpoints produits par le projet.

**Cas limite** : Des checkpoints anciens sauvés avec `torch.save(model.state_dict(), path)` (sans wrapper dict) fonctionnent aussi avec `weights_only=True`.

### Modifications

Remplacer systématiquement `weights_only=False` par `weights_only=True` dans les 6 fichiers :

#### `packages/ml/surrogate_filter.py` ligne 70

```python
# Avant :
checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)

# Après :
checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
```

#### `packages/ml/ppo.py` ligne 98

```python
# Avant :
checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)

# Après :
checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
```

#### `packages/ml/ppo.py` ligne 161

```python
# Avant :
checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)

# Après :
checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
```

#### `packages/ml/surrogate.py` ligne 109

```python
# Avant :
checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)

# Après :
checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
```

#### `packages/ml/surrogate.py` ligne 210

```python
# Avant :
checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)

# Après :
checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
```

#### `scripts/benchmark.py` ligne 153

```python
# Avant :
checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)

# Après :
checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=True)
```

### Test

Ajouter dans le test existant `tests/test_ml_surrogate.py` (ou créer si besoin) :

```python
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_surrogate_save_load_weights_only(tmp_path) -> None:
    """Verify checkpoints are loadable with weights_only=True."""
    import torch
    from packages.ml.surrogate import SurrogateModel

    model = SurrogateModel(input_dim=10, output_dim=4, hidden=32)
    path = tmp_path / "test_model.pt"
    model.save(path)

    # Must succeed with weights_only=True
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    assert "model" in checkpoint
    assert "config" in checkpoint
    assert checkpoint["version"] == "v1"
```

Ajouter un test similaire pour PPO :

```python
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_ppo_save_load_weights_only(tmp_path) -> None:
    """Verify PPO checkpoints are loadable with weights_only=True."""
    import torch
    from packages.ml.ppo import ActorCritic

    model = ActorCritic(obs_dim=16, act_dim=4, hidden=32)
    path = tmp_path / "test_ppo.pt"
    model.save(path)

    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    assert "model" in checkpoint
    assert checkpoint["version"] == "ppo_v2"
```

### Vérification grep

Après les modifications, cette commande ne doit retourner AUCUN résultat :

```bash
grep -rn "weights_only=False" packages/ scripts/
```

---

## TÂCHE 4 — Ajouter CORS middleware

### Problème actuel

L'API FastAPI n'a aucun middleware CORS. Si le dashboard Streamlit ou un frontend futur essaie d'appeler l'API depuis un domaine différent, les requêtes seront bloquées par le navigateur (Same-Origin Policy).

En production, un CORS trop permissif (`allow_origins=["*"]`) est un risque de sécurité (OWASP A05:2021). La configuration doit être restrictive par défaut avec une whitelist configurable.

### Modifications

#### 1. Ajouter les champs dans `packages/core/config.py`

```python
class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # ... champs existants ...
    cors_origins: str = ""      # <-- NOUVEAU : CSV séparé par virgule, ex: "http://localhost:8501,http://localhost:3000"
```

Et dans `get_settings()` :

```python
cors_origins=os.getenv("CORS_ORIGINS", ""),
```

#### 2. Modifier `apps/api/main.py`

```python
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from apps.api.routes.campaigns import router as campaigns_router
from apps.api.routes.candidates import router as candidates_router
from apps.api.routes.goals import router as goals_router
from apps.api.routes.health import router as health_router
from packages.core.config import get_settings
from packages.core.logging import get_logger


logger = get_logger(__name__)

settings = get_settings()

app = FastAPI(
    title="CryoSwarm-Q API",
    version="0.1.0",
    description=(
        "FastAPI backend for a hardware-aware multi-agent orchestration prototype "
        "for neutral-atom experimentation."
    ),
)

# ---- CORS middleware ----
_cors_origins: list[str] = [
    origin.strip()
    for origin in settings.cors_origins.split(",")
    if origin.strip()
]
if not _cors_origins:
    # Defaults safe for local development
    _cors_origins = ["http://localhost:8501", "http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

app.include_router(health_router)
app.include_router(goals_router)
app.include_router(campaigns_router)
app.include_router(candidates_router, prefix="/campaigns")


@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "Unhandled error on %s %s: %s",
        request.method,
        request.url.path,
        exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error."},
    )
```

**Points de sécurité** :
- `allow_origins` est une **whitelist explicite**, JAMAIS `["*"]`
- `allow_credentials=False` — pas de cookies cross-origin
- `allow_methods=["GET", "POST"]` — pas de PUT/DELETE/PATCH
- `allow_headers` inclut seulement `X-API-Key` et `Content-Type`

### Tests

Ajouter dans `tests/test_api_routes.py` ou créer `tests/test_cors.py` :

```python
"""CORS middleware tests."""
from __future__ import annotations

from fastapi.testclient import TestClient

from apps.api.main import app


def test_cors_allows_configured_origin() -> None:
    client = TestClient(app)
    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.headers.get("access-control-allow-origin") == "http://localhost:8501"


def test_cors_blocks_unknown_origin() -> None:
    client = TestClient(app)
    response = client.options(
        "/health",
        headers={
            "Origin": "http://evil.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    # Starlette CORS middleware omits the header for non-allowed origins
    assert "access-control-allow-origin" not in response.headers


def test_cors_does_not_allow_wildcard() -> None:
    client = TestClient(app)
    response = client.get(
        "/health",
        headers={"Origin": "http://localhost:8501"},
    )
    assert response.headers.get("access-control-allow-origin") != "*"


def test_cors_allows_api_key_header() -> None:
    client = TestClient(app)
    response = client.options(
        "/goals",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "X-API-Key",
        },
    )
    allowed_headers = response.headers.get("access-control-allow-headers", "").lower()
    assert "x-api-key" in allowed_headers
```

---

## TÂCHE 5 — Configurer timeout et pool MongoDB

### Problème actuel

**Fichier** : `packages/db/mongodb.py` (lignes 23-28)

Le `MongoClient` est créé sans aucun paramètre de timeout ni de pool :

```python
if _CLIENT is None:
    _CLIENT = MongoClient(settings.mongodb_uri)
```

Risques :
- **Timeout infini** : Une requête MongoDB qui ne répond jamais bloque le thread FastAPI indéfiniment (Denial of Service)
- **Pas de pool sizing** : Le comportement par défaut de PyMongo (`maxPoolSize=100`) peut ne pas être adapté
- **Pas de server selection timeout** : Si le cluster Atlas est en maintenance, l'app attend 30s par défaut avant de détecter le problème
- **Pas de threads safety considéré** : Le pattern global `_CLIENT` avec vérification `is None` n'est pas thread-safe (race condition sur `_CLIENT`)

### Modifications

#### 1. Ajouter les champs dans `packages/core/config.py` 

```python
class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # ... champs existants ...
    mongodb_connect_timeout_ms: int = 5000         # <-- NOUVEAU
    mongodb_server_selection_timeout_ms: int = 5000 # <-- NOUVEAU
    mongodb_socket_timeout_ms: int = 10000          # <-- NOUVEAU
    mongodb_max_pool_size: int = 20                 # <-- NOUVEAU
```

Et dans `get_settings()` :

```python
mongodb_connect_timeout_ms=int(os.getenv("MONGODB_CONNECT_TIMEOUT_MS", "5000")),
mongodb_server_selection_timeout_ms=int(os.getenv("MONGODB_SERVER_SELECTION_TIMEOUT_MS", "5000")),
mongodb_socket_timeout_ms=int(os.getenv("MONGODB_SOCKET_TIMEOUT_MS", "10000")),
mongodb_max_pool_size=int(os.getenv("MONGODB_MAX_POOL_SIZE", "20")),
```

#### 2. Modifier `packages/db/mongodb.py`

```python
from __future__ import annotations

import threading

from pymongo import MongoClient
from pymongo.database import Database

from packages.core.config import Settings, get_settings


COLLECTION_NAMES = [
    "experiment_goals",
    "register_candidates",
    "sequence_candidates",
    "robustness_reports",
    "campaigns",
    "agent_decisions",
    "memory",
    "evaluation_results",
]

_CLIENT: MongoClient | None = None
_CLIENT_LOCK = threading.Lock()


def get_mongo_client(settings: Settings | None = None) -> MongoClient:
    global _CLIENT
    settings = settings or get_settings()
    if not settings.mongodb_uri:
        raise RuntimeError("MONGODB_URI is not configured.")
    if _CLIENT is None:
        with _CLIENT_LOCK:
            # Double-checked locking pattern
            if _CLIENT is None:
                _CLIENT = MongoClient(
                    settings.mongodb_uri,
                    connectTimeoutMS=settings.mongodb_connect_timeout_ms,
                    serverSelectionTimeoutMS=settings.mongodb_server_selection_timeout_ms,
                    socketTimeoutMS=settings.mongodb_socket_timeout_ms,
                    maxPoolSize=settings.mongodb_max_pool_size,
                )
    return _CLIENT


def get_database(settings: Settings | None = None) -> Database:
    settings = settings or get_settings()
    client = get_mongo_client(settings)
    return client[settings.mongodb_db]


def close_mongo_client() -> None:
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is not None:
            _CLIENT.close()
            _CLIENT = None
```

**Points de sécurité** :
- `connectTimeoutMS=5000` : Empêche les connexions pendantes (DoS)
- `serverSelectionTimeoutMS=5000` : Fail-fast si le cluster est indisponible
- `socketTimeoutMS=10000` : Timeout sur les opérations individuelles
- `maxPoolSize=20` : Limite raisonnable pour un prototype FastAPI
- `threading.Lock()` : Thread-safety du singleton `_CLIENT` (CWE-362 — Race Condition)

### Tests

Ajouter dans `tests/test_db_repository.py` ou créer `tests/test_mongodb_config.py` :

```python
"""MongoDB configuration and timeout tests."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from packages.core.config import Settings


def test_default_timeouts_are_bounded() -> None:
    """Verify that default MongoDB timeouts are finite and reasonable."""
    settings = Settings(mongodb_uri="mongodb://fake")
    assert settings.mongodb_connect_timeout_ms > 0
    assert settings.mongodb_connect_timeout_ms <= 30000
    assert settings.mongodb_server_selection_timeout_ms > 0
    assert settings.mongodb_server_selection_timeout_ms <= 30000
    assert settings.mongodb_socket_timeout_ms > 0
    assert settings.mongodb_socket_timeout_ms <= 60000
    assert settings.mongodb_max_pool_size > 0
    assert settings.mongodb_max_pool_size <= 200


def test_mongo_client_passes_timeouts(monkeypatch) -> None:
    """Verify that get_mongo_client passes timeout settings to MongoClient."""
    import packages.db.mongodb as mongo_mod

    # Reset the global client
    monkeypatch.setattr(mongo_mod, "_CLIENT", None)

    captured_kwargs = {}

    class FakeMongoClient:
        def __init__(self, uri, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(mongo_mod, "MongoClient", FakeMongoClient)

    settings = Settings(
        mongodb_uri="mongodb://fake",
        mongodb_connect_timeout_ms=3000,
        mongodb_server_selection_timeout_ms=4000,
        mongodb_socket_timeout_ms=8000,
        mongodb_max_pool_size=15,
    )
    mongo_mod.get_mongo_client(settings)

    assert captured_kwargs["connectTimeoutMS"] == 3000
    assert captured_kwargs["serverSelectionTimeoutMS"] == 4000
    assert captured_kwargs["socketTimeoutMS"] == 8000
    assert captured_kwargs["maxPoolSize"] == 15

    # Cleanup
    monkeypatch.setattr(mongo_mod, "_CLIENT", None)


def test_close_mongo_client_is_idempotent(monkeypatch) -> None:
    """Verify that close_mongo_client can be called multiple times safely."""
    import packages.db.mongodb as mongo_mod

    monkeypatch.setattr(mongo_mod, "_CLIENT", None)
    mongo_mod.close_mongo_client()  # No-op when None
    mongo_mod.close_mongo_client()  # Still no-op
```

---

## TÂCHE 6 — Ne pas appeler `initialize_database()` à chaque requête

### Problème actuel

**Fichier** : `apps/api/dependencies.py` (lignes 10-14)

```python
def get_repository() -> CryoSwarmRepository:
    settings = get_settings()
    if not settings.has_mongodb:
        raise HTTPException(status_code=500, detail="MongoDB is not configured.")
    initialize_database()  # <-- Appelé à CHAQUE requête !
    return CryoSwarmRepository(settings)
```

Chaque requête HTTP appelle `initialize_database()`, qui :
1. Fait `database.list_collection_names()` — requête réseau vers MongoDB
2. Fait `database.create_collection()` pour chaque collection manquante
3. Crée 4 index par collection (8 collections × 4 = 32 `create_index` calls)
4. Log « MongoDB collections initialized. » à chaque fois

C'est :
- **Un gaspillage de bande passante** (dizaines de requêtes MongoDB par requête HTTP)
- **Un risque de latence** (chaque requête attend la réponse de MongoDB pour l'initialisation)
- **Un log polluant** (le message est répété à chaque appel)

### Solution

Utiliser un **startup event** FastAPI pour initialiser la base une seule fois au démarrage de l'application, et un **shutdown event** pour fermer proprement la connexion MongoDB.

### Modifications

#### 1. Modifier `apps/api/main.py`

Ajouter les lifecycle events :

```python
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from packages.core.config import get_settings
from packages.db.init_db import initialize_database
from packages.db.mongodb import close_mongo_client


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Application lifecycle: initialize DB on startup, close on shutdown."""
    init_settings = get_settings()
    if init_settings.has_mongodb:
        initialize_database()
        logger.info("MongoDB initialized during startup.")
    else:
        logger.warning("MongoDB not configured — skipping database initialization.")
    yield
    close_mongo_client()
    logger.info("MongoDB connection closed.")


app = FastAPI(
    title="CryoSwarm-Q API",
    version="0.1.0",
    description=(
        "FastAPI backend for a hardware-aware multi-agent orchestration prototype "
        "for neutral-atom experimentation."
    ),
    lifespan=lifespan,
)
```

#### 2. Modifier `apps/api/dependencies.py`

Retirer l'appel à `initialize_database()` :

```python
from __future__ import annotations

from fastapi import HTTPException

from packages.core.config import get_settings
from packages.db.repositories import CryoSwarmRepository


def get_repository() -> CryoSwarmRepository:
    """Provide a CryoSwarmRepository instance.

    Database initialization is handled once at application startup
    via the lifespan context manager in main.py.
    """
    settings = get_settings()
    if not settings.has_mongodb:
        raise HTTPException(status_code=500, detail="MongoDB is not configured.")
    return CryoSwarmRepository(settings)
```

#### 3. Ajouter un garde d'idempotence dans `init_db.py`

Même si l'appel est maintenant unique, ajouter un garde pour les cas d'appel direct (scripts, tests) :

```python
from __future__ import annotations

from packages.core.logging import get_logger
from packages.db.mongodb import COLLECTION_NAMES, get_database


logger = get_logger(__name__)

_INITIALIZED = False


def initialize_database() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return

    database = get_database()
    existing = set(database.list_collection_names())
    for collection_name in COLLECTION_NAMES:
        if collection_name not in existing:
            database.create_collection(collection_name)
        collection = database[collection_name]
        collection.create_index("id", unique=True)
        collection.create_index("created_at")
        collection.create_index("campaign_id")
        collection.create_index("goal_id")

    _INITIALIZED = True
    logger.info("MongoDB collections initialized.")


def reset_initialization_flag() -> None:
    """Reset the initialization flag (for testing only)."""
    global _INITIALIZED
    _INITIALIZED = False


if __name__ == "__main__":
    initialize_database()
```

#### 4. Mettre à jour le dashboard

`apps/dashboard/app.py` appelle aussi `initialize_database()` directement (ligne 111). Cela reste acceptable pour le dashboard Streamlit (qui est un process séparé), mais le garde d'idempotence empêche les appels multiples si Streamlit rerun le script.

### Tests

```python
def test_dependency_does_not_call_initialize_database(monkeypatch) -> None:
    """Verify get_repository() does not call initialize_database()."""
    import apps.api.dependencies as deps_mod
    from packages.core.config import Settings

    call_count = 0

    def _counting_init():
        nonlocal call_count
        call_count += 1

    monkeypatch.setattr(
        "packages.core.config.get_settings",
        lambda: Settings(mongodb_uri="mongodb://fake"),
    )
    # Ensure initialize_database is NOT imported or called in dependencies
    import inspect
    source = inspect.getsource(deps_mod.get_repository)
    assert "initialize_database" not in source


def test_init_db_idempotent(monkeypatch) -> None:
    """Verify initialize_database() only runs once."""
    import packages.db.init_db as init_mod

    init_mod.reset_initialization_flag()

    call_count = 0
    original = init_mod.get_database

    class FakeDatabase:
        def list_collection_names(self):
            nonlocal call_count
            call_count += 1
            return list(COLLECTION_NAMES)

        def __getitem__(self, name):
            return MagicMock()

    monkeypatch.setattr(init_mod, "get_database", lambda: FakeDatabase())

    init_mod.initialize_database()
    init_mod.initialize_database()
    init_mod.initialize_database()

    assert call_count == 1  # Only called once despite 3 invocations

    init_mod.reset_initialization_flag()
```

---

## ORDRE D'EXÉCUTION

1. **TÂCHE 5** — Timeout et pool MongoDB (fondation infrastructure)
2. **TÂCHE 6** — Singleton `initialize_database()` (dépend de l'infrastructure MongoDB)
3. **TÂCHE 4** — CORS middleware (modifie `main.py`, dépend des imports)
4. **TÂCHE 1** — Masquer les détails d'erreur (modifie `main.py`, `campaigns.py`, `health.py`)
5. **TÂCHE 2** — Authentification API key (dépend de config.py déjà modifié)
6. **TÂCHE 3** — `weights_only=True` dans torch.load (indépendant, rapide)

Après chaque tâche : `pytest tests/ -x --tb=short`

---

## CONTRAINTES FINALES

- **Zéro credential en dur.** Tous les secrets viennent de variables d'environnement.
- **Zéro `str(exc)` dans une réponse HTTP** en mode production.
- **Zéro `weights_only=False`** dans le codebase après la tâche 3.
- **Zéro `allow_origins=["*"]`** — CORS est une whitelist.
- **Zéro `# TODO` laissé dans le code produit.**
- **`from __future__ import annotations` en première ligne de chaque nouveau fichier.**
- **Chaque nouveau fichier a une docstring de module.**
- **Les imports sont groupés** : stdlib, third-party, project.
- **Pas de lignes > 120 caractères.**
- **Les logs utilisent le format `%s`** (lazy formatting), jamais f-strings.
- **Si un test existant casse, corrige-le immédiatement** en préservant le comportement testé.
- **`secrets.compare_digest()`** pour toute comparaison de secrets.
- **Ne JAMAIS loguer les credentials/tokens** — utiliser `logger.debug("Connecting to MongoDB...")` pas `logger.debug(f"Connecting to {uri}")`.

## CHECKLIST DE SÉCURITÉ POST-IMPLÉMENTATION

```bash
# 1. Vérifier qu'aucun secret n'est en dur
grep -rn "password\|secret\|token" packages/ apps/ --include="*.py" | grep -v "\.pyc" | grep -v "getenv\|environ\|os\." | grep -v "Field\|model_config"

# 2. Vérifier qu'aucun weights_only=False ne subsiste
grep -rn "weights_only=False" packages/ scripts/

# 3. Vérifier qu'aucun str(exc) n'est dans une réponse HTTP
grep -rn "str(exc)" apps/api/ --include="*.py"

# 4. Vérifier CORS
grep -rn "allow_origins" apps/ --include="*.py"

# 5. Tous les tests
pytest tests/ -v --tb=short

# 6. Vérification de sécurité spécifique
pytest tests/test_api_auth.py tests/test_cors.py tests/test_api_error_handling.py -v
```

Tous les tests doivent passer. Aucune régression tolérée.
