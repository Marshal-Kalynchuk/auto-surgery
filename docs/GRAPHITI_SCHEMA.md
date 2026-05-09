# Graphiti Memory Schema: Auto-Surgery Engineering Log

This document defines the structure and schema for our Graphiti-backed persistent engineering log. It ensures that all AI agents and developers record decisions, experiments, and architectural anchors in a uniform, queryable format.

## 1. Grouping Strategy

All memory episodes for this repository MUST use a single, unified `group_id`:
**`group_id: "auto-surgery"`**

A single group ensures that the knowledge graph can form edges between an experiment, the architectural component it validates, and the decision that follows.

## 2. Episode Kinds

Every episode added to Graphiti via `add_memory` (with `source="json"`) must conform to one of three `kind`s:

1. **`decision`**: Architectural Decision Records (ADRs). Used when a path is chosen or rejected.
2. **`experiment`**: Results from validation experiments (e.g., Hopfield capacity, IDM transfer).
3. **`doc_anchor`**: Core architectural concepts. Used to seed the graph with ground-truth definitions so decisions and experiments have canonical nodes to link to.

## 3. JSON Schemas (`episode_body`)

When calling `add_memory`, the `episode_body` must be a stringified JSON object matching the schemas below.

### A. Decision (ADR)
Used to log architectural choices, trade-offs, and superseded decisions.

```json
{
  "schema_version": "auto-surgery-log-1",
  "kind": "decision",
  "id": "ADR-<YYYYMMDD>-<short-name>",
  "title": "Why we chose <X>",
  "status": "accepted", 
  "date_utc": "YYYY-MM-DD",
  "context": {
    "problem": "Description of the problem being solved",
    "scope": ["<module-or-system>"]
  },
  "decision": {
    "chosen": "The accepted approach",
    "alternatives_considered": [
      { "name": "Alternative A", "rejected_because": "Reason" }
    ]
  },
  "consequences": {
    "pros": ["..."],
    "cons": ["..."]
  },
  "supersedes": ["ADR-<old-id>"],
  "links": {
    "primary": [{"type": "doc", "uri": "docs/ARCHITECTURE.md#..."}]
  }
}
```

### B. Experiment
Used to log validation runs, hypotheses, and outcomes.

```json
{
  "schema_version": "auto-surgery-log-1",
  "kind": "experiment",
  "id": "EXP-<YYYYMMDD>-<short-name>",
  "title": "Experiment: <Name>",
  "status": "completed",
  "date_utc": "YYYY-MM-DD",
  "hypothesis": "What we expected to happen",
  "target_risk": "RISK-<short-name>",
  "setup": {
    "inputs": ["Dataset or conditions"],
    "commands": ["Commands run"]
  },
  "results": {
    "metrics": {"key": "value"},
    "observations": ["What was seen"]
  },
  "outcome": {
    "supports_or_refutes": "supports | refutes | inconclusive",
    "conclusion": "What this means for the architecture"
  },
  "links": {
    "primary": [{"type": "doc", "uri": "docs/RISKS_AND_VALIDATION.md#..."}]
  }
}
```

### C. Doc Anchor
Used to seed the graph with canonical components and concepts.

```json
{
  "schema_version": "auto-surgery-log-1",
  "kind": "doc_anchor",
  "id": "DOC-<short-name>",
  "title": "Concept: <Name>",
  "definition": "What this is and what it does in the system",
  "responsibilities": ["..."],
  "links": {
    "primary": [{"type": "doc", "uri": "docs/..."}]
  }
}
```

## 4. Entity & Edge Conventions

Graphiti automatically extracts entities and edges from the JSON. To help it build a clean graph:
- **Use consistent IDs:** Always reference components by their exact names (e.g., "IDM transfer pipeline", "Hopfield memory").
- **Explicit target risks:** In experiments, explicitly name the risk being tested (e.g., "Occlusion handling risk") so Graphiti creates a `VALIDATES` or `TESTS` edge between the experiment and the risk.

## 5. Agent Workflow

1. **Adding Memory:** When an agent concludes an experiment or makes a load-bearing decision, it must call `add_memory` with `source="json"` and the stringified JSON matching the schemas above.
2. **Retrieval:** Agents should use `search_nodes` or `search_memory_facts` with `group_ids=["auto-surgery"]` to recall past context before proposing architectural changes.
3. **Superseding:** If a new decision invalidates an old one, the agent must include the old ID in the `supersedes` array of the new decision.
