# Agent Orchestration Platform

A multi-agent orchestration platform. Specialised AI agents work through customer-support cases
in chat-driven workflows. A message sent to `POST /api/chat` is routed to a YAML-defined
pipeline. A coordinator then dispatches it through specialised agents (**diagnostic →
resolution → summary**) and returns a structured response.

* **Skills.** Each agent calls reusable *skills*: Python modules, discovered dynamically, behind
  typed contracts.
* **Structured output.** Every agent output is validated against the agent's JSON schema, and
  invalid output is repaired, retried or replaced.
* **Shared context.** Agents reason over each other's findings through a Redis-backed context
  store, across stages and across chat turns.
* **Platform.** JWT authentication with role-based access, structured JSON logs, Prometheus
  metrics and health checks.

The platform runs **without an API key**. The default LLM provider is a mock that returns
plausible, schema-valid JSON. One setting switches to any OpenAI-compatible endpoint.

## Architecture

```
  client ──► POST /auth/token ──► JWT (sub, roles)
     │
     └─► POST /api/chat {message, case_id?} ──► JWTAuthMiddleware ──► RBAC (operator+)
                                                         │
                                                         ▼
 ┌──────────────────────── Orchestrator (engine.py) ─────────────────────────────┐
 │ PipelineRouter: requested / route rules (regex, has_history) / default        │
 │                                                                               │
 │  for each stage in pipeline YAML:                                             │
 │    read case context from Redis ──► `when` condition ──► SkillRouter          │
 │    ──► skills (contract-validated, concurrent) ──► input mapping ──► Agent    │
 │    ──► LLM (mock | OpenAI-compatible) ──► parse + JSON-schema validate        │
 │        └─ invalid: repair prompt ─► stage retries ─► fallback agent           │
 │                    ─► default output ─► skip | abort                          │
 │    ──► write validated output to Redis (findings)                             │
 └───────────────┬───────────────────────────────────────┬───────────────────────┘
                 ▼                                       ▼
      Redis shared context store            Skill registry (dynamic discovery)
      case:{id} meta, turns, findings,      builtin package, extra packages,
      skills; per-case lock; TTL            entry points, plugins/skills/*.py

 /health (Redis, loaded components)   /metrics (Prometheus)   JSON logs (request_id, case_id)
```

**Pipelines shipped** (`config/pipelines/`):

| Pipeline | Routing | Stages |
|---|---|---|
| `case_resolution` | Default | `diagnose` (diagnostic) → `resolve` (resolution, skipped while the diagnosis needs more info) → `summarize` (summary) |
| `case_followup` | The case has history *and* the message says the fix didn't work ("still", "didn't work", "again", ...) | `resolve` (re-plans from the earlier diagnosis, excluding steps already tried) → `summarize` |

**Agents** (`config/agents/`):

| Agent | Role | Skills it may use |
|---|---|---|
| `diagnostic` | Category, severity, ranked root-cause hypotheses with evidence | `knowledge_base_search`, `log_analyzer`, `customer_lookup` |
| `diagnostic_lite` | Fallback for `diagnostic`. It `extends` the diagnostic agent, so it has the same contracts, with a simpler prompt | (same) |
| `resolution` | Ordered action plan with owners, escalation decision and SLA | `knowledge_base_search`, `escalation_policy` |
| `summary` | Customer reply, adapted to sentiment, plus an internal note and the case status | `sentiment_analyzer` |

**Skills:**

| Skill | Source | Contract (input → output) |
|---|---|---|
| `knowledge_base_search` | built-in | query, top_k → ranked articles with symptoms and steps (`data/knowledge_base.json`) |
| `log_analyzer` | built-in | text → HTTP error statuses, exception names, error codes, failure signals |
| `customer_lookup` | built-in | text or customer id → plan, region, ticket history (deterministic CRM stub) |
| `escalation_policy` | built-in | severity, category, plan, previous attempts → escalate?, team, SLA hours |
| `sentiment_analyzer` | **plugin** (`plugins/skills/`) | text → sentiment, score, frustration signals |

## How the pieces work

### Integration contracts

Integration contracts live in `orchestrator/contracts.py`.

* **Skills.** A skill is a class with pydantic `input_model` and `output_model` and an async
  `run`. The orchestrator validates arguments before the call and the result after it, so a
  skill can't hand an agent malformed data.
* **Agents.** An agent is a YAML spec: a role prompt, an `input_schema`, an `output_schema`
  (JSON Schema) and its `allowed_skills`.
* **Discovering them.** `GET /api/contracts` lists every agent and skill contract.
* **Adding something new needs no core changes:**
  * A new skill is a `.py` file dropped into `plugins/skills/`, an importable package listed in
    `ORCH_SKILL_PACKAGES`, or an installed distribution exposing the `agent_orchestrator.skills`
    entry point.
  * A new agent or pipeline is a YAML file.
  * `POST /api/admin/reload` picks up either without a restart.

### Structured output parsing

`orchestrator/structured.py` handles parsing and validation.

**Parsing.** The parser takes JSON out of a fenced code block, or the first balanced `{...}` in
the text.

**Validation.** The result is checked against the agent's `output_schema` with JSON Schema
2020-12.

**Repair.** If validation fails, the agent sends the exact errors back to the model ("`hypotheses/0:
'confidence' is a required property`") and asks for corrected JSON, for up to `max_repairs`
rounds.

**Stage failure policy.** If the agent still fails, the stage's `on_failure` policy is applied
in this order:

1. `retries`: run the stage again.
2. `fallback_agent`: a different agent that must have the same output contract. This is checked
   at load time.
3. `default_output`: a static result, validated against the schema at load time.
4. `otherwise`: `skip` the stage or `abort` the pipeline.

**Transforming data between stages.** Stage `inputs` and skill `args` are mapping templates over
the shared context:

* `$.stages.diagnose.severity` and `"$.previous.diagnose.hypotheses[0].hypothesis"` are paths
  into the context.
* `{$coalesce: [...]}` takes the first non-null value.
* `{$count: $.history}` counts the items in a list.

The rendered inputs are validated against the next agent's `input_schema` before it runs.

### Skill routing

Each stage binds skills, and each binding has optional conditions and argument mappings:

```yaml
skills:
  - skill: log_analyzer
    args: {text: $.message}
    when: {matches: 'error|exception|timeout|\b[45]\d\d\b'}   # only when there is error evidence
  - skill: customer_lookup
    args: {text: $.message}
    when: {matches: 'CUST-\d+'}
```

The skill router runs every binding whose condition holds, concurrently, through the skill
contracts. A skill that fails is recorded and the stage carries on, unless the binding is marked
`required: true`. A pipeline may only bind skills that appear in the agent's `allowed_skills`;
this is checked when the pipeline loads.

### Multi-agent reasoning over shared context

Each stage writes its validated output to Redis as soon as it finishes. Each stage builds its
inputs from a **fresh read** of the store:

* `$.stages.*`: findings from earlier stages in this turn.
* `$.previous.*`: findings from earlier turns.
* `$.history`: earlier messages and replies.

So the resolution agent reasons over the diagnostic agent's hypotheses and severity. On a
follow-up turn ("it still doesn't work") it re-plans from the stored diagnosis, excluding steps
already tried, and the escalation policy counts the earlier attempts.

A per-case Redis lock (`SET NX` with an owner token) serialises turns on the same case. Keys
expire after `ORCH_CONTEXT_TTL_SECONDS` (default 24 h). `GET /api/cases/{id}` returns the whole
shared context.

### Authentication and RBAC

* **Getting a token.** `POST /auth/token` checks a username and password against
  `config/users.yaml` (PBKDF2-SHA256 hashes) and returns an HS256 JWT. The token carries `sub`
  and `roles`, plus `iss`, `iat` and `exp`.
* **Checking it.** `JWTAuthMiddleware` authenticates every `/api/*` request: signature, expiry,
  issuer, and that the roles are valid. Otherwise it returns 401.
* **Authorising the call.** Endpoints declare a minimum role with `require_role(...)`. The role
  hierarchy is `viewer` < `operator` < `admin`.

| Endpoint | Minimum role |
|---|---|
| `POST /api/chat` | operator |
| `GET /api/cases/{id}`, `/api/pipelines`, `/api/agents`, `/api/skills`, `/api/contracts` | viewer |
| `POST /api/admin/reload` | admin |
| `POST /auth/token`, `GET /health`, `GET /metrics` | public |

Demo users: `admin`/`admin-pass` (admin), `alice`/`alice-pass` (operator) and `bob`/`bob-pass`
(viewer). Set `ORCH_JWT_SECRET` in any real deployment.

### Observability

**Logs.** JSON lines on stdout. Each line carries `request_id` (taken from `X-Request-ID` or
generated) and `case_id`, plus structured fields. Logged events include:

* `request`
* `pipeline selected`
* `stage finished` (agent, status, attempts, LLM calls, skills, duration)
* `agent output failed validation`
* `skill failed`
* `pipeline finished`

**Metrics.** `GET /metrics` exposes Prometheus metrics:

| Metric | Labels |
|---|---|
| `http_requests_total` | method, route, status |
| `http_request_duration_seconds` (histogram) | method, route |
| `pipeline_runs_total` | pipeline, status |
| `pipeline_duration_seconds` (histogram) | pipeline |
| `agent_runs_total` | agent, status |
| `agent_duration_seconds` (histogram) | agent |
| `skill_invocations_total` | skill, status |
| `llm_calls_total` | provider, status |
| `structured_output_validation_failures_total` | agent |
| `pipeline_fallbacks_total` | pipeline, stage, strategy |
| `auth_failures_total` | reason |

**Health.** `GET /health` pings Redis (503 if it's down) and lists the loaded pipelines, agents,
skills and LLM provider.

## Run

### Docker Compose (app + Redis)

```bash
docker compose up -d --build
curl localhost:8000/health
open http://localhost:8000/docs
```

### Chat with it

```bash
TOKEN=$(curl -s -X POST localhost:8000/auth/token -H 'content-type: application/json' \
  -d '{"username":"alice","password":"alice-pass"}' | jq -r .access_token)

# New case -> case_resolution pipeline
curl -s -X POST localhost:8000/api/chat -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"message":"Since this morning none of our users can log in with SSO - Okta shows invalid SAML response and we get HTTP 403. Account CUST-1042."}' | jq

# Same case, fix didn't work -> case_followup pipeline (reasons over the stored diagnosis)
curl -s -X POST localhost:8000/api/chat -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"case_id":"<case_id from above>","message":"We re-uploaded the metadata but it still does not work!"}' | jq

# Shared context of the case (viewer role is enough)
curl -s localhost:8000/api/cases/<case_id> -H "Authorization: Bearer $TOKEN" | jq
```

### Locally

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
docker compose up -d redis
uvicorn orchestrator.api.app:app --reload
```

### Configuration

All settings are environment variables with the `ORCH_` prefix:

| Variable | Default | Meaning |
|---|---|---|
| `ORCH_REDIS_URL` | `redis://localhost:6379/0` | Shared context store |
| `ORCH_CONTEXT_TTL_SECONDS` | `86400` | Case context expiry |
| `ORCH_LLM_PROVIDER` | `mock` | `mock` or `openai` (any OpenAI-compatible `/chat/completions`) |
| `ORCH_LLM_BASE_URL`, `ORCH_LLM_API_KEY`, `ORCH_LLM_MODEL` | `https://api.openai.com/v1`, empty, `gpt-4o-mini` | Real model endpoint (OpenAI, vLLM, Ollama, LiteLLM, ...) |
| `ORCH_LLM_RESPONSE_FORMAT` | `json_object` | `json_schema` (strict structured outputs), `json_object`, or `none` |
| `ORCH_MOCK_INVALID_RATE` | `0` | Mock only: fraction of calls returning malformed output (to exercise repair and fallbacks) |
| `ORCH_JWT_SECRET`, `ORCH_JWT_TTL_MINUTES` | dev secret, `60` | Token signing |
| `ORCH_AGENTS_DIR`, `ORCH_PIPELINES_DIR`, `ORCH_SKILL_PLUGIN_DIR`, `ORCH_SKILL_PACKAGES` | `config/agents`, `config/pipelines`, `plugins/skills`, empty | Where agents, pipelines and skills come from |

## Results

These come from running the compose stack with the mock LLM on a laptop. They show the platform
working end to end. The mock's reasoning is deterministic heuristics, so these are not
measurements of model quality.

### Response to a new SSO case

The message was: *"none of our users can log in with SSO … HTTP 403. Account CUST-1042"*.

```json
{
  "case_id": "case-20261111-879e77c5",
  "turn": 1,
  "pipeline": "case_resolution",
  "routed_by": "default",
  "status": "completed",
  "case_status": "escalated",
  "reply": "Thanks for reaching out. It looks like this is related to: sso login fails with invalid saml response. On our side, we are escalating to our identity-platform team and will update you within 1 hour.",
  "stages": [
    {"id": "diagnose", "agent": "diagnostic", "status": "completed", "attempts": 1, "llm_calls": 1,
     "skills": ["knowledge_base_search", "log_analyzer", "customer_lookup"],
     "output": {"category": "authentication", "severity": "critical", "affected_component": "identity-provider",
                "hypotheses": [{"hypothesis": "SSO login fails with invalid SAML response", "confidence": 0.77,
                                "evidence": ["matches KB-1001 symptom: users cannot log in with SSO",
                                             "HTTP 403 in report", "signal: auth_failure"], "kb_article": "KB-1001"}, "..."],
                "needs_more_info": false, "customer_plan": "enterprise", "...": "..."}},
    {"id": "resolve", "agent": "resolution", "status": "completed",
     "skills": ["knowledge_base_search", "escalation_policy"],
     "output": {"plan_type": "escalation", "escalate": true, "escalation_team": "identity-platform", "sla_hours": 1,
                "steps": ["4 KB-1001 support steps", "escalate to identity-platform"], "...": "..."}},
    {"id": "summarize", "agent": "summary", "status": "completed", "skills": ["sentiment_analyzer"],
     "output": {"status": "escalated", "next_action": "Hand off to identity-platform",
                "tags": ["authentication", "critical", "escalated"], "...": "..."}}
  ],
  "duration_ms": 5.5
}
```

The output is abridged (`"..."`). Reading it:

* **Diagnose.** The skill router ran `log_analyzer` because of the "HTTP 403", and
  `customer_lookup` because of the CUST id.
* **Severity.** The diagnosis became *critical* because the outage affects every user and the
  customer is on the enterprise plan.
* **Escalation.** The escalation policy turned that into the identity-platform team with a
  1-hour SLA.

### Scenarios exercised

| Scenario | Result |
|---|---|
| New SSO outage (enterprise customer) | `case_resolution`: diagnose → resolve → summarize; critical, escalated to identity-platform, 1 h SLA |
| Same case: "re-uploaded the metadata but it still doesn't work!" | Routed to `case_followup` (route match on history and "still"). It reused the turn-1 diagnosis from Redis, repeated none of the 4 steps already tried, escalated, and apologised because the sentiment skill detected frustration. |
| Vague message ("it is broken") | Diagnosis set `needs_more_info`, so `resolve` was **skipped** by its `when` condition. The reply asks 3 clarifying questions; case status `needs_info`. |
| Billing double charge | Matched KB-2002; in-progress plan with a 24 h SLA |
| `ORCH_MOCK_INVALID_RATE=0.5`, 20 cases | 61 invalid outputs rejected by validation. 15 pipelines completed (13 stages fixed by the repair prompt), 4 degraded (1 fallback agent, 4 default outputs), 1 failed |
| `ORCH_MOCK_INVALID_RATE=0.7`, 30 cases | 160 invalid outputs rejected. 12 completed, 15 degraded, 3 failed (see below) |
| No token / bad token / viewer calling chat / viewer calling reload | 401 / 401 / 403 / 403; an admin reload re-discovered all components |
| `ORCH_LLM_PROVIDER=openai` against a fake OpenAI-compatible server | Bearer key, model and `json_schema` response format sent correctly; the pipeline completed |
| `ORCH_LLM_PROVIDER=openai` with the endpoint unreachable | The stage recorded `ConnectError` and the pipeline returned `status: failed` with stage errors, without crashing |

What happened in the 30-case run at `ORCH_MOCK_INVALID_RATE=0.7`:

| Strategy | Times applied |
|---|---|
| `diagnostic_lite` fallback agent answered | 7 |
| Summary default output used | 10 |
| Resolve stage skipped | 4 |
| Pipeline aborted | 3 |

All of these are counted in `pipeline_fallbacks_total` and `structured_output_validation_failures_total`.

## Project layout

```
src/orchestrator/
  api/            FastAPI app, routes, request/response schemas
  llm/            provider interface, OpenAI-compatible client, mock provider
  skills/         registry + discovery; builtin/ skills
  agents.py       YAML agent specs, LLM agent with validation and repair loop
  pipeline.py     pipeline specs, conditions, fallback policies, router
  engine.py       orchestrator: stages, skill routing, fallbacks, context writes
  context.py      Redis shared context store
  structured.py   JSON extraction, schema validation, mapping templates
  contracts.py    skill and agent integration contracts
  auth.py         JWT middleware, RBAC, user store
  observability.py JSON logging, Prometheus metrics
config/agents/    agent YAML       config/pipelines/  pipeline YAML     config/users.yaml
plugins/skills/   drop-in skills   data/              knowledge base
```

## CI/CD

`.github/workflows/ci.yml` runs on every push and pull request:

1. **Lint and type check**:
   * `ruff check` and `ruff format --check`.
   * `mypy --strict` over `src` and the plugins.
   * Loads every agent, pipeline and skill, which validates all YAML references and schemas.
2. **Build** (runs after step 1 passes):
   * Builds the wheel and the Docker image.
   * Starts the compose stack.
   * Smoke-tests it: health, a 401 without a token, a full chat pipeline completing, and the
     metrics endpoint.

## Tech stack

Python 3.12, FastAPI, Pydantic v2, Redis (redis-py asyncio), JSON Schema (`jsonschema`), PyJWT,
httpx, prometheus-client, PyYAML, Docker and Docker Compose, GitHub Actions, ruff and mypy.
