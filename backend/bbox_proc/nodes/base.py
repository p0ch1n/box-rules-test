"""Abstract base class for all pipeline nodes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

from bbox_proc.schema.models import NodeConfig
from bbox_proc.spatial.geometry import BBox


class PortType(str, Enum):
    """Wire type system — only compatible port types may be connected."""

    BoxStream = "BoxStream"
    """Single-branch stream of BBox objects."""

    Collection = "Collection"
    """Multi-branch aggregate of BBox objects (with lineage metadata)."""

    LogicSignal = "LogicSignal"
    """Boolean trigger with attached metadata dict."""

    ImageStream = "ImageStream"
    """Stream of raw images (numpy arrays) fed into detection nodes."""

    AnnotatedStream = "AnnotatedStream"
    """Stream of AnnotatedFrames — each frame carries an image plus its BBoxes."""

    ReferenceImageStream = "ReferenceImageStream"
    """Static reference images that do not change frame-to-frame.

    Used for configuration-time images such as template-matching references or
    background models. A node that outputs this type is a *self-seeding* source:
    it declares no input ports and loads its data from its own config on first
    execute(). The scheduler will NOT inject pipeline input into such nodes.
    """


@dataclass(frozen=True)
class PortDefinition:
    """Describes a single input or output port on a node."""

    name: str
    port_type: PortType
    description: str = ""
    optional: bool = False


class BaseNode(ABC):
    """Abstract base for all processing nodes.

    Subclasses must:
    - Declare `input_ports` and `output_ports`.
    - Implement `execute(inputs)` returning a typed output dict.
    """

    def __init__(self, config: NodeConfig) -> None:
        self.config = config
        self.node_id = config.id
        self._parsed_config = config.parse_config()

    # ------------------------------------------------------------------ #
    # Port declarations (must be implemented by subclasses)
    # ------------------------------------------------------------------ #

    @property
    @abstractmethod
    def input_ports(self) -> List[PortDefinition]:
        """Ordered list of input port definitions."""
        ...

    @property
    @abstractmethod
    def output_ports(self) -> List[PortDefinition]:
        """Ordered list of output port definitions."""
        ...

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #

    @abstractmethod
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the node logic.

        Args:
            inputs: mapping of port_name → value (List[BBox] or bool+metadata).

        Returns:
            mapping of port_name → output value.
        """
        ...

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _get_bboxes(self, inputs: Dict[str, Any], port: str) -> List[BBox]:
        """Safely extract a BBox list from inputs, defaulting to []."""
        return inputs.get(port, [])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.node_id!r})"
