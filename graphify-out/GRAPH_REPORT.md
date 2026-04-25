# Graph Report - C:\Users\zayem\Desktop\urban_heat_env  (2026-04-24)

## Corpus Check
- 6 files · ~6,229 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 49 nodes · 95 edges · 7 communities detected
- Extraction: 71% EXTRACTED · 29% INFERRED · 0% AMBIGUOUS · INFERRED: 28 edges (avg confidence: 0.64)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]

## God Nodes (most connected - your core abstractions)
1. `CityGrid` - 18 edges
2. `TaskResult` - 9 edges
3. `main()` - 7 edges
4. `CellState` - 6 edges
5. `ActiveIntervention` - 6 edges
6. `Proposal` - 6 edges
7. `CityState` - 6 edges
8. `Core Urban Heat Island environment logic.  The 8x8 city grid holds cells with su` - 6 edges
9. `Initialise/reinitialise the grid deterministically.` - 6 edges
10. `main()` - 4 edges

## Surprising Connections (you probably didn't know these)
- `CityGrid` --uses--> `CellState`  [INFERRED]
  C:\Users\zayem\Desktop\urban_heat_env\server\environment.py → C:\Users\zayem\Desktop\urban_heat_env\models.py
- `CityGrid` --uses--> `ActiveIntervention`  [INFERRED]
  C:\Users\zayem\Desktop\urban_heat_env\server\environment.py → C:\Users\zayem\Desktop\urban_heat_env\models.py
- `CityGrid` --uses--> `Proposal`  [INFERRED]
  C:\Users\zayem\Desktop\urban_heat_env\server\environment.py → C:\Users\zayem\Desktop\urban_heat_env\models.py
- `CityGrid` --uses--> `CityState`  [INFERRED]
  C:\Users\zayem\Desktop\urban_heat_env\server\environment.py → C:\Users\zayem\Desktop\urban_heat_env\models.py
- `Core Urban Heat Island environment logic.  The 8x8 city grid holds cells with su` --uses--> `TaskResult`  [INFERRED]
  C:\Users\zayem\Desktop\urban_heat_env\server\environment.py → C:\Users\zayem\Desktop\urban_heat_env\models.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.4
Nodes (10): BaseModel, Core Urban Heat Island environment logic.  The 8x8 city grid holds cells with su, Initialise/reinitialise the grid deterministically., ActionRequest, ActiveIntervention, CellState, CityState, ObsResult (+2 more)

### Community 1 - "Community 1"
Cohesion: 0.36
Nodes (2): CityGrid, TaskResult

### Community 2 - "Community 2"
Cohesion: 0.46
Nodes (7): format_prompt(), get_state(), get_tasks(), grade_task(), main(), reset_env(), step_env()

### Community 3 - "Community 3"
Cohesion: 0.29
Nodes (5): get_tasks(), grade(), reset(), state(), step()

### Community 4 - "Community 4"
Cohesion: 0.47
Nodes (5): env_step(), format_env_prompt(), main(), Hackathon Round 2: Minimal TRL Training Script This demonstrates how to train th, Hits the local OpenEnv API exactly like inference.py

### Community 5 - "Community 5"
Cohesion: 0.5
Nodes (0): 

### Community 6 - "Community 6"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **2 isolated node(s):** `Hackathon Round 2: Minimal TRL Training Script This demonstrates how to train th`, `Hits the local OpenEnv API exactly like inference.py`
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 6`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CityGrid` connect `Community 1` to `Community 0`, `Community 5`?**
  _High betweenness centrality (0.271) - this node is a cross-community bridge._
- **Why does `main()` connect `Community 4` to `Community 5`?**
  _High betweenness centrality (0.152) - this node is a cross-community bridge._
- **Why does `state()` connect `Community 3` to `Community 5`?**
  _High betweenness centrality (0.092) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `CityGrid` (e.g. with `CellState` and `CityState`) actually correct?**
  _`CityGrid` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `TaskResult` (e.g. with `CityGrid` and `Core Urban Heat Island environment logic.  The 8x8 city grid holds cells with su`) actually correct?**
  _`TaskResult` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `CellState` (e.g. with `CityGrid` and `Core Urban Heat Island environment logic.  The 8x8 city grid holds cells with su`) actually correct?**
  _`CellState` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Hackathon Round 2: Minimal TRL Training Script This demonstrates how to train th`, `Hits the local OpenEnv API exactly like inference.py` to the rest of the system?**
  _2 weakly-connected nodes found - possible documentation gaps or missing edges._