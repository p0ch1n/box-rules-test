# Node Developer Guide

This guide explains how to add a new processing node to rule-engine. The system is designed so that adding a node requires changes only inside well-defined boundaries — the framework discovers new types automatically.

---

## Table of Contents

1. [Port Type System](#1-port-type-system)
2. [Node Categories](#2-node-categories)
3. [Adding a Backend Node](#3-adding-a-backend-node)
4. [Adding a Frontend Node](#4-adding-a-frontend-node)
5. [Common Patterns](#5-common-patterns)
   - [Frame-by-frame Processing Node](#51-frame-by-frame-processing-node)
   - [Self-seeding Reference Image Node](#52-self-seeding-reference-image-node)
   - [Template Matching Node (end-to-end example)](#53-template-matching-node-end-to-end-example)
6. [Schema and Serialization](#6-schema-and-serialization)
7. [Testing Requirements](#7-testing-requirements)
8. [Checklist](#8-checklist)

---

## 1. Port Type System

Every connection in the pipeline carries a typed value. Connecting ports of incompatible types is rejected both at canvas editing time (frontend) and validated implicitly at runtime (backend).

### Port Types

| Type | Backend enum | Frontend enum | Color | Carries |
|------|-------------|---------------|-------|---------|
| `BoxStream` | `PortType.BoxStream` | `PortType.BoxStream` | `#3b82f6` blue | `List[BBox]` |
| `Collection` | `PortType.Collection` | `PortType.Collection` | `#f59e0b` amber | `List[BBox]` with lineage metadata |
| `LogicSignal` | `PortType.LogicSignal` | `PortType.LogicSignal` | `#22c55e` green | `bool` + metadata dict |
| `ImageStream` | `PortType.ImageStream` | `PortType.ImageStream` | `#7c3aed` purple | `List[np.ndarray]` — per-frame pipeline input |
| `AnnotatedStream` | `PortType.AnnotatedStream` | `PortType.AnnotatedStream` | `#f97316` orange | `List[AnnotatedFrame]` — image + bbox pairs |
| `ReferenceImageStream` | `PortType.ReferenceImageStream` | `PortType.ReferenceImageStream` | `#e11d48` rose | `List[np.ndarray]` — static config-time images |

### Key Distinction: ImageStream vs ReferenceImageStream

| | `ImageStream` | `ReferenceImageStream` |
|---|---|---|
| **Changes per frame?** | Yes | No |
| **Source** | `pipeline.execute_frame(images=[...])` call | Node's own config (file path, base64, etc.) |
| **Typical node** | `DetectionNode` | `TemplateLoaderNode`, `BackgroundModelNode` |
| **Scheduler behavior** | Injected by scheduler on every run | Node loads once, caches in `self._ref` |

### Compatibility Rules

A source port can only connect to a target port of the same type, with one exception:

```
BoxStream  →  BoxStream   ✓
BoxStream  →  Collection  ✓  (single-stream shortcut to LogicNode)
Collection →  Collection  ✓
LogicSignal → LogicSignal ✓
ImageStream → ImageStream ✓
AnnotatedStream → AnnotatedStream ✓
ReferenceImageStream → ReferenceImageStream ✓
```

All cross-type connections are rejected.

### Port Colors — Always Use the Canonical Map

Never hardcode color hex strings in node components. Import `PORT_TYPE_COLORS`:

```typescript
import { PORT_TYPE_COLORS, PortType } from '@/nodes/types'

// In your node component:
<Handle style={{ background: PORT_TYPE_COLORS[PortType.ReferenceImageStream] }} />
```

The map lives in `frontend/src/nodes/types.ts` and is the single source of truth.

---

## 2. Node Categories

Before implementing, decide which category your node falls into:

| Category | input_ports | Scheduler behavior | Example |
|---|---|---|---|
| **Pipeline-input source** | `[PortDefinition("input", PortType.ImageStream)]` | Injected with `images` from `execute_frame` | `DetectionNode` |
| **BBox-input source** | `[PortDefinition("input", PortType.BoxStream)]` | Injected with `input_bboxes` from `execute_frame` | legacy source nodes |
| **Self-seeding source** | `[]` (empty list) | Not injected — receives `{}`, loads own data | `TemplateLoaderNode` |
| **Processing node** | one or more ports | Receives upstream outputs via edges | `FilterNode`, `ImageAnalysisNode` |

---

## 3. Adding a Backend Node

### Step 1: Create the node file

Create `backend/bbox_proc/nodes/your_node.py`.

```python
"""YourNode — one-line description."""

from __future__ import annotations

from typing import Any, Dict, List

from bbox_proc.nodes.base import BaseNode, PortDefinition, PortType
from bbox_proc.nodes.registry import NodeRegistry
from bbox_proc.schema.models import NodeConfig


@NodeRegistry.register("your_type")   # must match JSON schema "type" value
class YourNode(BaseNode):

    @property
    def input_ports(self) -> List[PortDefinition]:
        return [
            PortDefinition("input", PortType.BoxStream, "Incoming bounding boxes"),
        ]

    @property
    def output_ports(self) -> List[PortDefinition]:
        return [
            PortDefinition("output", PortType.BoxStream, "Filtered bounding boxes"),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        bboxes = self._get_bboxes(inputs, "input")
        result = [b for b in bboxes if self._passes(b)]
        return {"output": result}

    def _passes(self, bbox) -> bool:
        ...
```

### Step 2: Register the import

Add one line to `backend/bbox_proc/nodes/__init__.py`:

```python
from bbox_proc.nodes import your_node as _your_node  # noqa: F401
```

That is all. `NodeRegistry.create(config)` will find the new type automatically.

### Parsed Config Pattern

Parse the raw `config` dict into a typed dataclass in `__init__`:

```python
from dataclasses import dataclass

@dataclass
class YourNodeConfig:
    threshold: float = 0.5

class YourNode(BaseNode):
    def __init__(self, config: NodeConfig) -> None:
        super().__init__(config)
        raw = config.config or {}
        self._cfg = YourNodeConfig(
            threshold=float(raw.get("threshold", 0.5)),
        )
```

### BBox is Immutable

`BBox` is a frozen dataclass. All spatial transforms return new instances. Never modify a BBox in place.

### Python 3.9 Compatibility

Use `List[T]` and `Dict[K, V]` from `typing`, not `list[T]` or `dict[K, V]`:

```python
# Correct
from typing import Any, Dict, List
def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]: ...

# Wrong (Python 3.10+ only)
def execute(self, inputs: dict[str, Any]) -> dict[str, Any]: ...
```

---

## 4. Adding a Frontend Node

### Step 1: Create the node directory

```
frontend/src/nodes/YourNode/
├── definition.ts    # port declarations + defaultConfig
└── index.tsx        # React component
```

### Step 2: definition.ts

```typescript
import type { NodeDefinition } from '../types'
import { PortType } from '../types'
import { YourNodeComponent } from './index'

export const YourNodeDefinition: NodeDefinition = {
  type: 'your_type',           // must match backend @NodeRegistry.register value
  label: 'Your Node',
  description: 'One-line description shown in the canvas sidebar',
  inputPorts: [
    {
      name: 'input',
      portType: PortType.BoxStream,
      label: 'BBoxes',
      description: 'Incoming bounding boxes',
    },
  ],
  outputPorts: [
    {
      name: 'output',
      portType: PortType.BoxStream,
      label: 'Filtered',
      description: 'Filtered bounding boxes',
    },
  ],
  defaultConfig: {
    threshold: 0.5,
  },
  component: YourNodeComponent,
}
```

### Step 3: index.tsx

```typescript
import { Handle, Position } from 'reactflow'
import { PORT_TYPE_COLORS, PortType } from '@/nodes/types'
import type { NodeComponentProps } from '@/nodes/types'
import { registerNodeType } from '@/nodes/registry'
import { YourNodeDefinition } from './definition'

registerNodeType(YourNodeDefinition)

export function YourNodeComponent({ id, data, selected }: NodeComponentProps) {
  return (
    <div style={{ border: selected ? '2px solid #3b82f6' : '2px solid #e5e7eb', borderRadius: 8, padding: 12, background: '#fff', minWidth: 180 }}>
      {/* Input handle */}
      <Handle
        type="target"
        position={Position.Left}
        id="input"
        style={{ background: PORT_TYPE_COLORS[PortType.BoxStream], width: 12, height: 12 }}
      />

      <div style={{ fontWeight: 600, marginBottom: 8 }}>{data.label}</div>

      {/* Config fields */}
      <label style={{ fontSize: 12 }}>
        Threshold
        <input type="number" defaultValue={data.config.threshold as number} style={{ width: '100%' }} />
      </label>

      {/* Output handle */}
      <Handle
        type="source"
        position={Position.Right}
        id="output"
        style={{ background: PORT_TYPE_COLORS[PortType.BoxStream], width: 12, height: 12 }}
      />
    </div>
  )
}
```

### Step 4: Register the import

Add one line to `frontend/src/nodes/index.ts`:

```typescript
import './YourNode'
```

---

## 5. Common Patterns

### 5.1 Frame-by-frame Processing Node

Receives `AnnotatedStream` (image + bboxes), does per-bbox ROI analysis, outputs filtered result.

```python
def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
    frames: List[AnnotatedFrame] = inputs.get("input", [])
    out_frames: List[AnnotatedFrame] = []
    out_bboxes: List[BBox] = []

    for frame in frames:
        survivors = [b for b in frame.bboxes if self._analyze_roi(frame.image, b)]
        if survivors:
            out_frames.append(frame.with_bboxes(survivors))   # shares image ref, no copy
            out_bboxes.extend(survivors)

    return {"output": out_frames, "bboxes": out_bboxes}
```

`AnnotatedFrame.with_bboxes()` returns a new frame sharing the same image array (zero copy).

### 5.2 Self-seeding Reference Image Node

A node that loads a static reference image from config (e.g., a template for template matching). It declares **no input ports** and caches the loaded image across frames.

```python
@NodeRegistry.register("template_loader")
class TemplateLoaderNode(BaseNode):

    def __init__(self, config: NodeConfig) -> None:
        super().__init__(config)
        raw = config.config or {}
        self._path: str = str(raw.get("image_path", ""))
        self._cache: Optional[List[np.ndarray]] = None   # loaded on first execute

    @property
    def input_ports(self) -> List[PortDefinition]:
        return []   # ← empty: self-seeding, scheduler will NOT inject pipeline input

    @property
    def output_ports(self) -> List[PortDefinition]:
        return [
            PortDefinition("output", PortType.ReferenceImageStream, "Template image"),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self._cache is None:
            import cv2
            img = cv2.imread(self._path)
            if img is None:
                raise RuntimeError(f"TemplateLoaderNode: cannot read image at {self._path!r}")
            self._cache = [img]
        return {"output": self._cache}
```

**Why `input_ports = []` matters**: The scheduler only injects per-frame data into nodes that explicitly declare input ports. With an empty list the node is left alone — `execute({})` is called and the node provides its own data.

### 5.3 Template Matching Node (end-to-end example)

A node that receives both target frames (`AnnotatedStream`) and a reference template (`ReferenceImageStream`), filters bboxes whose ROI matches the template above a threshold.

**Backend:**

```python
@NodeRegistry.register("template_match")
class TemplateMatchNode(BaseNode):

    @property
    def input_ports(self) -> List[PortDefinition]:
        return [
            PortDefinition("input",    PortType.AnnotatedStream,     "Target frames"),
            PortDefinition("template", PortType.ReferenceImageStream, "Template image"),
        ]

    @property
    def output_ports(self) -> List[PortDefinition]:
        return [
            PortDefinition("output", PortType.AnnotatedStream, "Matched frames"),
            PortDefinition("bboxes", PortType.BoxStream,       "Matched bboxes (flat)"),
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        frames:    List[AnnotatedFrame] = inputs.get("input",    [])
        templates: List[np.ndarray]     = inputs.get("template", [])

        if not templates:
            return {"output": [], "bboxes": []}

        template = templates[0]   # use first reference image
        cfg = self._cfg

        out_frames, out_bboxes = [], []
        for frame in frames:
            survivors = [
                b for b in frame.bboxes
                if self._match_score(frame.image, b, template) >= cfg.threshold
            ]
            if survivors:
                out_frames.append(frame.with_bboxes(survivors))
                out_bboxes.extend(survivors)

        return {"output": out_frames, "bboxes": out_bboxes}
```

**Canvas wiring:**

```
TemplateLoaderNode ──(ReferenceImageStream)──► TemplateMatchNode.template
DetectionNode.annotated ─(AnnotatedStream)───► TemplateMatchNode.input
TemplateMatchNode.bboxes ──(BoxStream)────────► LogicNode
```

**Frontend definition:**

```typescript
inputPorts: [
  { name: 'input',    portType: PortType.AnnotatedStream,     label: 'Frames'    },
  { name: 'template', portType: PortType.ReferenceImageStream, label: 'Template'  },
],
outputPorts: [
  { name: 'output', portType: PortType.AnnotatedStream, label: 'Matched frames' },
  { name: 'bboxes', portType: PortType.BoxStream,       label: 'Matched BBoxes' },
],
```

---

## 6. Schema and Serialization

`schema/pipeline.schema.json` lists valid node `type` strings. When you add a new node type, add it to the `"type"` enum in the schema so that exported pipelines validate correctly:

```json
"type": {
  "type": "string",
  "enum": ["filter", "logic", "relation", "merge", "detection", "image_analysis", "template_loader", "template_match"]
}
```

The schema does **not** encode port types — type safety is enforced in application code only.

---

## 7. Testing Requirements

Minimum 80% line coverage. Write tests in `backend/tests/test_your_node.py`.

### Required test cases

1. **Happy path** — node processes valid input and returns expected output.
2. **Empty input** — `execute({})` or `execute({"input": []})` returns `{"output": []}` without crashing.
3. **Config parsing** — config defaults are applied correctly; invalid values raise early.
4. **Immutability** — output BBoxes are not the same objects as input BBoxes when transforms are applied.

### Self-seeding node tests

For reference image loader nodes, mock the filesystem call (`cv2.imread`) and verify:
- Cache is populated on first call.
- Second call does NOT re-read the file (cached).
- Missing file raises `RuntimeError` with a useful message.

### Example test skeleton

```python
import pytest
from bbox_proc.nodes.your_node import YourNode
from bbox_proc.schema.models import NodeConfig
from bbox_proc.spatial.geometry import BBox

def _make_node(config: dict | None = None) -> YourNode:
    nc = NodeConfig(id="test", type="your_type", position={"x": 0, "y": 0}, config=config or {})
    return YourNode(nc)

def test_empty_input_returns_empty():
    node = _make_node()
    result = node.execute({"input": []})
    assert result == {"output": []}

def test_filters_correctly():
    node = _make_node({"threshold": 0.5})
    bbox = BBox(x=0, y=0, w=10, h=10, confidence=0.8, class_name="cat")
    result = node.execute({"input": [bbox]})
    assert len(result["output"]) == 1
```

---

## 8. Checklist

Before opening a PR for a new node:

**Backend**
- [ ] `@NodeRegistry.register("type_string")` decorator present
- [ ] `input_ports` and `output_ports` declared correctly
- [ ] `execute()` handles empty inputs gracefully
- [ ] Config parsed to typed dataclass in `__init__`
- [ ] `BBox` objects never mutated — transforms return new instances
- [ ] `List[T]` / `Dict[K, V]` used (Python 3.9 compat)
- [ ] Import added to `bbox_proc/nodes/__init__.py`
- [ ] Tests written and coverage ≥ 80%

**Frontend**
- [ ] `definition.ts` created with correct `type` string
- [ ] `registerNodeType()` called in `index.tsx` on module load
- [ ] Port handles use `PORT_TYPE_COLORS[PortType.X]` — no raw hex strings
- [ ] Import added to `src/nodes/index.ts`
- [ ] TypeScript: `npm run type-check` passes

**Schema**
- [ ] New `type` string added to `"type"` enum in `schema/pipeline.schema.json`
