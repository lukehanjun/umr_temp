# UMR — Onboarding Guide

Welcome. Read this document to understand what we're building, see a working demo, and know exactly what you need to implement.

Read `CLAUDE.md` alongside this — it has the full technical spec. This doc focuses on the big picture and your tasks.

---

## What We're Building

We want to teach a robot arm to perform tabletop manipulation tasks (e.g., "stack the red cube on the blue cylinder"). Rather than training an end-to-end model, we use a **visual subgoal pipeline**: we break the task down into a sequence of intermediate goal images, and a separate low-level policy model (developed by another team) figures out the motor commands to reach each one.

Our system has four components:

```
┌─────────────────────────────────────────────────────────────────┐
│                        This Repo                                │
│                                                                  │
│  scene photo + task description                                  │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────┐     "stack black marker,   ┌────────────┐  │
│  │  Goal Planner   │ ──► then red, then green" ► │  Explorer  │  │
│  └─────────────────┘     (goal frames)           └────────────┘  │
│           │                                          │           │
│           ▼                          generates variants + new tasks
│  ┌─────────────────┐                                             │
│  │    Verifier     │ ◄── actual camera photo                     │
│  └─────────────────┘                                             │
│           │  advance / retry / replan                            │
└───────────┼──────────────────────────────────────────────────────┘
            ▼
  ┌───────────────────────┐
  │  Policy Model (other  │  goal frame → motor commands
  │  team's repo)         │  returns: actual observation
  └───────────────────────┘
```

---

## A Worked Example

The `data/` folder contains a real example. The task is:

> **"Stack the three markers into a vertical tower so they stand upright on the table"**

**Initial scene** (`data/input.jpg`):

Three Expo markers (black, red, green) lying flat on a round wooden table, alongside a remote, tissue box, coffee cup, and other objects.

**Human decomposition** (`data/human/`):

A human performed the task and we photographed each intermediate state. This is the ground truth we're aiming for:

| Step | Photo | What happened |
|------|-------|---------------|
| Step 1 | `human/step1.jpg` | Markers grouped together, beginning to be positioned |
| Step 2 | `human/step2.jpg` | Black + red markers standing upright, stacked (2-tower) |
| Step 3 | `human/step3.jpg` | All three markers stacked into a complete upright tower (green on top) |

**AI-generated goal frames** (`data/output_frames/`):

Our planner already runs end-to-end and produces synthesized goal frames. The reasoning trace is in `output_frames/reasoning_trace.txt`. The planner (Claude Opus 4.6) correctly deduced the 3-step decomposition:

1. Black marker standing upright, cap-end down
2. Red marker balanced upright on top of the black marker
3. Green marker completing the tower on top of the red marker

The generated images are edited from the initial photo — the same table, same lighting, same background objects — with only the markers changed. **Comparing the AI frames to the human photos shows exactly where the system is strong and where it needs improvement** — which is what the verifier and explorer will help us measure and fix.

---

## What Works Right Now

You can run the planner today:

```bash
# Setup
uv sync
cp .env.example .env
# Fill in: OPENROUTER_API_KEY, OPENAI_API_KEY

# Run on the included example
uv run python scripts/run_episode.py
# Or point to your own image + task:
uv run python scripts/run_episode.py path/to/image.jpg path/to/task.txt
```

Output is saved to `data/output_frames/`:
- `step_01.jpg`, `step_02.jpg`, ... — synthesized goal frame images
- `reasoning_trace.txt` — full chain-of-thought + per-step descriptions

---

## What You Need to Build

There are four areas. Three are direct implementations; one (simulation) is foundational infrastructure.

---

### 1. Verifier

**What it does**: After the policy executes a step, we get a camera photo back. The verifier compares this photo to the intended goal frame and decides: did we succeed, should we retry, or should we replan from scratch?

**Files**:
- `sharpa/api/vlm.py` — the VLM API call (implement first)
- `sharpa/agents/verifier.py` — the scoring + decision logic
- `tests/test_verifier.py` — extend these tests once implemented

**How it works**:
1. Send both images to a VLM (GPT-4o) with a judge prompt: did the actual image satisfy the goal state? The VLM returns a binary yes/no and a natural language explanation of what matched or didn't.
2. Decide:
   - `success = true` → **advance** to next goal frame
   - `success = false`, retries remaining → **retry**
   - `success = false`, retries exhausted → **replan** (send back to planner with current state)

`max_retries` comes from `settings` — never hardcode it.

Thresholds come from `settings` — never hardcode numbers.

See the inline `# TODO` comments in each file for step-by-step implementation guidance.

---

### 2. Explorer

The explorer has two distinct jobs:

#### 2a. Alternative Decompositions

Given a goal frame that the verifier is struggling with, generate K alternative ways to achieve the same intermediate state. For example, if "stack black marker first" keeps failing, maybe try "stack green marker first" as an alternative approach.

The explorer generates these variants — it does not decide when to use them. That logic is in `loop.py`.

**File**: `sharpa/agents/explorer.py`

#### 2b. New Task Generation

Beyond helping with the current episode, the explorer should generate **new tasks and new task variants** to build a training dataset for the policy model. Examples:
- "Stack the markers in reverse order (green first)"
- "Arrange the markers horizontally side by side"
- "Place only the black and red markers upright, leave green flat"

These new tasks feed back into the planner + verifier loop to generate more labelled (initial state, goal frames, verification outcomes) data.

This is the more open-ended part of the explorer. Think about: what makes a good training task? How do you ensure variety? How do you avoid tasks that are physically impossible?

**File**: `sharpa/agents/explorer.py` (same file, separate method or extended `generate_variants`)

---

### 3. Orchestration Loop

**What it does**: Ties the planner, verifier, and explorer together into a full episode — from initial image to task success or failure.

**File**: `sharpa/loop.py`

The loop is the central coordinator. See the `# TODO` comment in `loop.py` for a detailed pseudocode sketch of the intended control flow (plan → execute → verify → retry/replan/advance → explore probabilistically).

Also implement `sharpa/memory/store.py` alongside this — the loop needs to persist episodes to SQLite so we can review them and export goal frames for the policy team.

---

### 4. Robot Simulation

**Why this exists**: The policy model team will eventually consume our goal frames and return real camera observations. But while we're developing, we need something to run against. The `StubPolicyBridge` (already implemented) just returns a grey image — good enough for unit tests, but not useful for actually evaluating whether the verifier and planner are working well together.

**What to build**: A lightweight simulation environment that:
- Takes a goal frame image as input
- Returns a synthetic "actual observation" that realistically simulates what a robot might observe after attempting to achieve that goal
- Supports variable success rates (sometimes it almost gets there, sometimes it fails badly) so the verifier can exercise all three decision paths

This does **not** need physics simulation. A reasonable approach:
- On "success": return the goal frame image with minor perturbations (small random crops, brightness jitter, slight rotation) to simulate imperfect execution
- On "partial success": blend between the previous state and the goal frame
- On "failure": return the previous state with no change

The key output is a PIL image with realistic noise, not a perfect copy of the goal frame. This gives the verifier something meaningful to score.

**File**: `sharpa/interface/policy_bridge.py` — implement `SimulatedPolicyBridge` as a third implementation of `PolicyBridgeBase` (alongside the existing `StubPolicyBridge` and `RealPolicyBridge`).

---

## Task Dependencies

```
vlm.py (Task 1a)
    └── verifier.py (Task 1b)
            └── loop.py (Task 3)
                    └── store.py (Task 3b)

explorer.py (Task 2)  ──────────────► loop.py (Task 3)

SimulatedPolicyBridge (Task 4) ──────► loop.py (Task 3)
```

Tasks 1a, 2, and 4 can be started in parallel. Task 3 (loop) should be last — it needs the others.

---

## Codebase Map

```
sharpa/
├── config.py             Settings loaded from .env (API keys, model names, thresholds)
├── models/episode.py     Data models: GoalFrame, VerificationResult, Episode
│
├── api/
│   ├── llm.py            LLM wrapper (OpenRouter + OpenAI routing)          ✓ done
│   ├── imagegen.py       Image gen/edit wrapper (Gemini + DALL-E)            ✓ done
│   └── vlm.py            VLM scoring wrapper                                 ✗ TODO
│
├── agents/
│   ├── planner.py        GoalPlannerAgent — task → goal frames               ✓ done
│   ├── verifier.py       VerifierAgent — actual vs goal scoring              ✗ TODO
│   └── explorer.py       ExplorerAgent — variants + new task generation      ✗ TODO
│
├── loop.py               Episode orchestration loop                          ✗ TODO
│
├── memory/
│   └── store.py          SQLite persistence for episodes                     ✗ TODO
│
└── interface/
    └── policy_bridge.py  StubPolicyBridge ✓ done | SimulatedPolicyBridge ✗ TODO
```

---

## Coding Conventions

- `uv add <pkg>` for new dependencies — never `pip install`
- All agent methods are `async def` — always use `await`
- `settings` is always injected into constructors — never imported at module level inside agents
- `structlog` for all logging — no `print()`. INFO for decisions, DEBUG for scores
- Never hardcode thresholds, model names, or API keys — always `settings.*`
- Mock all `api/` wrappers in tests — no real HTTP calls in the test suite

See `CLAUDE.md` section 9 for the full standards. See `sharpa/agents/planner.py` for a complete example of a working agent, and `tests/test_planner.py` for how to test it.
