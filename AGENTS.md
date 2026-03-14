# SHARPA — Reasoning & Planning Layer
## Codex Project Instructions

> **Scope**: This repo owns the **reasoning and planning layer only**.  
> The low-level policy (converting goal frames → motor commands) is maintained by a separate team.  
> Our interface to them is: we **output goal frames**, they **consume them**.

---

## 1. What This System Does

Given:
- An initial camera image of the scene
- A high-level task description (e.g. "stack the red cube on the blue cylinder")

This system produces:
- A sequence of **intermediate goal frames** — synthesized images representing desired intermediate robot states
- **Reasoning traces** explaining why each subgoal was chosen
- **Verification decisions** (advance / retry / replan) based on comparing the actual robot observation against the intended goal frame
- **Novel goal variants** generated through exploration, for improving coverage

We do **not** control the robot. We do **not** train the policy. We provide the goal frames and the high-level coordination logic. The policy team consumes our output via a defined interface.

---

## 2. Repository Structure

```
sharpa/
├── AGENTS.md                  # ← You are here
├── README.md
├── pyproject.toml             # uv-managed, Python 3.11+
├── .env.example
│
├── sharpa/
│   ├── __init__.py
│   ├── config.py              # Pydantic settings from .env
│   │
│   ├── agents/
│   │   ├── planner.py         # GoalPlannerAgent — core of this repo
│   │   ├── verifier.py        # VerifierAgent — VLM-as-judge
│   │   └── explorer.py        # ExplorerAgent — goal variant generation
│   │
│   ├── loop.py                # Orchestration: ties agents together per episode
│   │
│   ├── models/                # Pydantic data models only — no logic
│   │   └── episode.py         # GoalFrame, VerificationResult, Episode
│   │
│   ├── api/                   # Wrappers around external LLM/VLM/ImGen APIs
│   │   ├── llm.py             # call_llm() → str
│   │   ├── imagegen.py        # generate_image() → PIL.Image
│   │   └── vlm.py             # score_image_pair() → dict[str, float]
│   │
│   ├── memory/
│   │   └── store.py           # SQLite store for episodes + verification results
│   │
│   └── interface/
│       └── policy_bridge.py   # THE BOUNDARY: how we hand off goal frames to the policy team
│
├── scripts/
│   ├── run_episode.py         # Simulate a full episode (stub robot observations)
│   └── export_goals.py        # Export goal frames for policy team consumption
│
└── tests/
    ├── test_planner.py
    ├── test_verifier.py
    ├── test_explorer.py
    └── test_loop.py
```

---

## 3. System Boundary

This is the most important thing to understand about scope:

```
  ┌─────────────────────────────────────────────────────┐
  │              THIS REPO (Reasoning Layer)             │
  │                                                      │
  │   initial_image + task_description                   │
  │           │                                          │
  │           ▼                                          │
  │   GoalPlannerAgent                                   │
  │     - LLM CoT reasoning                              │
  │     - Image generation (goal frame synthesis)        │
  │           │                                          │
  │           │  goal_frames: list[GoalFrame]            │
  │           ▼                                          │
  │   ┌──────────────────────────────┐                  │
  │   │      Coordination Loop       │ ◄── actual_image │
  │   │                              │  (from bridge)   │
  │   │  VerifierAgent               │                  │
  │   │   - VLM scoring              │                  │
  │   │   - advance/retry/replan     │                  │
  │   │                              │                  │
  │   │  ExplorerAgent               │                  │
  │   │   - variant generation       │                  │
  │   └──────────────────────────────┘                  │
  │           │                                          │
  │           │  GoalFrame (image + metadata)            │
  └───────────┼──────────────────────────────────────────┘
              │
              ▼  interface/policy_bridge.py
  ┌───────────────────────────────┐
  │     POLICY TEAM (other repo)  │
  │  goal frame → motor commands  │
  │  returns: actual observation  │
  └───────────────────────────────┘
```

**What we receive**: actual camera observations after each action step (via bridge).  
**What we send**: goal frames (image + description + step metadata).  
**The bridge**: `interface/policy_bridge.py` — one file, one interface, kept minimal.

---

## 4. Agent Specifications

### 4.1 GoalPlannerAgent (`agents/planner.py`)

The core of this repo. Decomposes a task into a sequence of visual subgoals.

**Inputs**:
- `initial_image: PIL.Image`
- `task_description: str`
- `current_state_image: PIL.Image | None` — provided when replanning mid-episode
- `n_goals: int` — from config, typically 4–8
- `failed_frames: list[GoalFrame] | None` — passed during replan to avoid repeating failures

**Outputs**: `list[GoalFrame]`

**Process**:
1. Build a CoT prompt for the LLM with the scene image + task
2. LLM returns structured JSON with N subgoal descriptions and reasoning
3. For each subgoal, call `imagegen.generate_image()` to synthesize the goal frame image
4. Pack into `GoalFrame` objects with reasoning trace attached

**Required LLM output format** (enforce via system prompt):
```xml
<thinking>
  [spatial reasoning, object states, hand kinematics constraints,
   what needs to be true at each step for the task to succeed...]
</thinking>
<goals>
  [
    {
      "step": 1,
      "description": "...",        // used as image gen prompt
      "key_visual_features": [],   // what the verifier should check
      "confidence": 0.0-1.0        // planner's self-assessed confidence
    }
  ]
</goals>
```

**Replanning**: when `current_state_image` is provided, the prompt includes a diff — "this is where we are now vs where we planned to be" — and `failed_frames` are included as negative examples to avoid.

**Preferred model**: `Codex-opus-4-6` (strong spatial CoT). Configurable via settings.

---

### 4.2 VerifierAgent (`agents/verifier.py`)

Judges whether the actual observation matches the intended goal frame.

**Inputs**:
- `goal_frame: GoalFrame`
- `actual_image: PIL.Image`
- `retry_count: int`

**Outputs**: `VerificationResult`

**Scoring**: Calls the VLM via `api/vlm.py` with a structured judge prompt asking for three scores:
- `semantic_score`: does the actual image match the described subgoal state?
- `pose_score`: are the geometric/positional features correct?
- `contact_score`: is the grasp/contact quality appropriate?

**Decision rule**:
```python
composite = 0.5 * semantic + 0.3 * pose + 0.2 * contact
if composite >= settings.advance_threshold:   decision = "advance"
elif retry_count < settings.max_retries:      decision = "retry"
else:                                          decision = "replan"
```

Thresholds are always from `settings` — never hardcoded.

---

### 4.3 ExplorerAgent (`agents/explorer.py`)

Generates alternative goal frame variants for under-explored subgoal states.

**Inputs**:
- `goal_frame: GoalFrame` — the current goal being attempted
- `episode: Episode` — full episode context for conditioning
- `k: int` — number of variants to generate (default 3)

**Outputs**: `list[GoalFrame]` (variants, not replacements for the original)

**Process**:
1. Prompt LLM: "given this subgoal and scene, generate K alternative visual descriptions that could achieve the same intermediate state via different approaches"
2. Synthesize images for each variant description
3. Return variants — **Explorer does not execute anything**

Loop.py decides when to invoke Explorer (based on `epsilon`) and what to do with variants. Execution and logging is loop.py's responsibility, not Explorer's.

---

## 5. Data Models (`models/episode.py`)

```python
class GoalFrame(BaseModel):
    step: int
    description: str
    key_visual_features: list[str]
    reasoning: str                  # LLM CoT trace — always preserved
    image: PIL.Image
    confidence: float
    is_variant: bool = False        # True if generated by ExplorerAgent

class VerificationResult(BaseModel):
    goal_frame: GoalFrame
    actual_image: PIL.Image
    semantic_score: float
    pose_score: float
    contact_score: float
    composite_score: float
    decision: Literal["advance", "retry", "replan"]
    reasoning: str
    retry_count: int

class Episode(BaseModel):
    id: str                         # UUID
    task_description: str
    initial_image: PIL.Image
    goal_frames: list[GoalFrame]
    verifications: list[VerificationResult]
    success: bool | None = None     # None = in progress
    created_at: datetime
```

Images stored as base64 in SQLite, decoded to PIL on read. No image files written to disk.

---

## 6. The Policy Bridge (`interface/policy_bridge.py`)

The only place where we interact with the policy team's system.

```python
class PolicyBridgeBase(ABC):
    """Abstract interface. Two implementations: StubPolicyBridge and RealPolicyBridge."""

    async def send_goal_frame(self, goal_frame: GoalFrame) -> None:
        """Send a goal frame to the policy execution system."""
        ...

    async def get_observation(self) -> PIL.Image:
        """Get the current camera observation after the last action."""
        ...

    async def is_step_complete(self) -> bool:
        """Check if the policy has finished attempting the current goal."""
        ...
```

**Rules**:
- Keep it minimal — it is a seam, not a feature
- `StubPolicyBridge` returns synthetic images; used in all tests and dev runs
- `RealPolicyBridge` is filled in during hardware integration
- Policy team concerns must never leak past this file into agents

---

## 7. Configuration (`config.py`)

```python
class Settings(BaseSettings):
    anthropic_api_key: str
    openai_api_key: str

    planner_llm: str = "Codex-opus-4-6"
    verifier_vlm: str = "gpt-4o"
    imagegen_model: str = "dall-e-3"

    advance_threshold: float = 0.80
    max_retries: int = 3
    n_goals: int = 6
    epsilon_explore: float = 0.20

    db_path: str = "sharpa_memory.db"
```

No hardcoded values. Always `settings.*`.

---

## 8. API Wrapper Contracts

All external calls go through these three wrappers. Agents never call httpx or SDKs directly.

```python
# api/llm.py
async def call_llm(
    messages: list[dict],
    model: str,
    system: str | None = None,
    response_format: Literal["text", "json"] = "text",
    max_tokens: int = 2048,
) -> str: ...

# api/imagegen.py
async def generate_image(
    prompt: str,
    model: str,
    size: tuple[int, int] = (512, 512),
    reference_image: PIL.Image | None = None,
) -> PIL.Image: ...

# api/vlm.py
async def score_image_pair(
    goal_image: PIL.Image,
    actual_image: PIL.Image,
    task_description: str,
    model: str,
) -> dict[str, float]:  # keys: "semantic", "pose", "contact"
    ...
```

---

## 9. Coding Standards

- **Python 3.11+**, type hints everywhere, no bare `Any`
- **`uv`** for packages — `uv add <pkg>`, never `pip install`
- **`ruff`** for formatting and linting
- **`asyncio`** throughout — all agent methods are `async def`
- **`structlog`** for logging — no `print()`. INFO for decisions, DEBUG for scores.
- **`pytest` + `pytest-mock`** — mock all `api/` wrappers; never make real HTTP calls in tests
- Agents return results with `error: str | None`, never raise — loop handles errors
- `settings` injected into constructors, not imported at module level

---

## 10. Custom Slash Commands

### `/plan-goals`
```
Run GoalPlannerAgent and display generated goal frames.
Args: <task_description>

1. Instantiate GoalPlannerAgent
2. Call plan() with a stub initial image
3. Print each GoalFrame: step, description, reasoning, confidence
```

### `/verify-pair`
```
Score a goal/actual image pair via VerifierAgent.
Args: <goal_image_path> <actual_image_path> <goal_description>

1. Load both images
2. Run VerifierAgent.score()
3. Show: semantic, pose, contact, composite, decision, reasoning
```

### `/explore-variants`
```
Generate alternative goal frames for a subgoal.
Args: <goal_description> [--k N]

1. Build a stub GoalFrame
2. Run ExplorerAgent.generate_variants()
3. Print all K variants with descriptions and confidence
```

### `/run-stub-episode`
```
Run a full episode with StubPolicyBridge (no robot needed).
Args: <task_description>

1. Use StubPolicyBridge returning synthetic observations
2. Run loop.py end-to-end
3. Print full trace: goal frames, each verification result, outcome
```

### `/add-tests`
```
Scaffold pytest tests for a module.
Args: <module_path>

1. Read the module
2. Write tests/test_<module>.py
3. Mock all api/ wrappers — no real API calls
4. Run pytest to confirm they pass
```

---

## 11. What Is Out of Scope

Do not build or modify:
- Motor control, joint targets, ROS2 nodes
- LoRA fine-tuning of the policy
- Simulation environments (MuJoCo, Isaac)
- Anything past `interface/policy_bridge.py`

If you find yourself touching robot control code, stop.

---

## 12. Glossary

| Term | Meaning |
|---|---|
| Goal Frame | Synthesized image of a desired intermediate robot state, with reasoning |
| Episode | One full run from initial state to task success or failure |
| Verification | VLM-based comparison: actual observation vs intended goal frame |
| Replan | Calling GoalPlannerAgent again mid-episode from current state |
| Variant | Alternative goal frame for the same subgoal, from ExplorerAgent |
| Policy Bridge | Single interface file between this repo and the policy team |
| CoT | Chain-of-thought reasoning trace, always stored in GoalFrame |