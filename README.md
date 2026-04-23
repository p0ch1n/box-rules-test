# rule-engine

Real-time bounding box processing pipeline — visual editor (React + Vite) + Python execution engine.

## Architecture

```
rule-engine/
├── frontend/   # React 18 + Vite + React-Flow canvas editor
├── backend/    # Python 3.9+ execution engine (pure library)
├── schema/     # Shared JSON Schema (Draft-07)
└── docs/       # spec.md
```

### Design Principles

- **Registry Pattern**: adding a new node type requires zero changes to the framework.
- **Immutable data**: `BBox` is frozen; all transformations return new instances.
- **Pure library engine**: `Pipeline.execute_frame(bboxes)` — no side effects; caller decides what to do with results.
- **Single-frame processing**: stateless by design, safe for concurrent frame processing.

---

## Backend

### Install

```bash
cd backend
pip install -e ".[dev]"
```

### Run tests

```bash
pytest tests/ -v
```

Coverage: **93%** (requirement: 80%).

### Run a pipeline exported from the frontend

```bash
# Run the bundled example (uses built-in fake detections)
python examples/run_pipeline.py

# Run with your own exported pipeline file
python examples/run_pipeline.py my_pipeline.json
```

Full annotated source: [`backend/examples/run_pipeline.py`](backend/examples/run_pipeline.py)

### API usage

```python
from bbox_proc.engine.interpreter import Interpreter
from bbox_proc.spatial.geometry import BBox

# ── Load ──────────────────────────────────────────────────────────────────
pipeline = Interpreter.load("pipeline.json")        # from file
# pipeline = Interpreter.from_json(json_string)     # from string (API / stream)

# ── Execute a frame ───────────────────────────────────────────────────────
bboxes = [
    BBox(x=100, y=50, w=80,  h=180, confidence=0.92, class_name="person"),
    BBox(x=300, y=200, w=150, h=90,  confidence=0.88, class_name="car"),
]
result = pipeline.execute_frame(bboxes)

# ── Read results ──────────────────────────────────────────────────────────
for node_id, signal in result.signals().items():
    if signal["triggered"]:
        print(f"ALERT [{node_id}] label={signal['label']!r}  counts={signal['class_counts']}")

# Get a specific node's output boxes
passed_boxes = result.get_node_output("filter-1", port="output")

# Full output map (all nodes, all ports)
all_outputs = result.all_outputs()
```

#### `BBox` fields

| Field | Type | Description |
|---|---|---|
| `x`, `y` | float | Top-left corner (pixels) |
| `w`, `h` | float | Width and height (pixels) |
| `confidence` | float | Detector confidence score [0, 1] |
| `class_name` | str | Object class label |
| `metadata` | dict | Optional key/value pairs (lineage, relation info, …) |

#### `ExecutionResult` methods

| Method | Returns | Description |
|---|---|---|
| `signals()` | `dict[node_id, signal]` | All Logic node trigger states |
| `get_node_output(id, port)` | any | Output of a specific node port |
| `all_outputs()` | `dict` | Full per-node output map |

### Extending with a new node type

```python
# my_node.py
from bbox_proc.nodes.base import BaseNode, PortDefinition, PortType
from bbox_proc.nodes.registry import NodeRegistry

@NodeRegistry.register("my_node")
class MyNode(BaseNode):
    @property
    def input_ports(self): ...
    @property
    def output_ports(self): ...
    def execute(self, inputs): ...
```

No other files need modification.

---

## Frontend

### Install & run

```bash
cd frontend
npm install
npm run dev       # http://localhost:5173
npm run build     # production bundle
npm run type-check
```

### Features

- Drag nodes from the left palette onto the canvas.
- Connect ports — type validation prevents invalid wires (BoxStream / Collection / LogicSignal).
- Class list auto-inferred from upstream FilterNodes.
- **Export JSON** — downloads `pipeline.json` matching the shared schema.
- **Import JSON** — reloads a saved pipeline from disk.

---

## JSON Schema

Located at `schema/pipeline.schema.json` (Draft-07). Shared between frontend validation and backend Pydantic models.

```json
{
  "version": "1.0",
  "metadata": { "tool_id": "cam-01", "class_list": ["person", "car"] },
  "nodes": [
    {
      "id": "filter-1", "type": "filter",
      "position": {"x": 100, "y": 100},
      "config": { "class_name": "person", "field": "confidence", "operator": "gt", "threshold": 0.7 }
    }
  ],
  "edges": [
    { "id": "e1", "source": "filter-1", "source_port": "output",
      "target": "merge-1", "target_port": "input_0" }
  ]
}
```

---

## Port Type System

| Type | Description | Connects to |
|---|---|---|
| `BoxStream` | Single-branch BBox stream | BoxStream, Collection |
| `Collection` | Multi-branch aggregate (with lineage) | Collection |
| `LogicSignal` | Boolean trigger + metadata | LogicSignal |

**FilterNode / RelationNode**: BoxStream → BoxStream
**MergeNode**: N × BoxStream → Collection
**LogicNode**: BoxStream or Collection → LogicSignal

> A `BoxStream` can connect directly to a `LogicNode` without going through a `MergeNode`.
