"""VLMNode — sends images to a Vision Language Model API and parses detections."""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, List

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from rule_execution_engine.nodes.base import BaseNode, PortDefinition, PortType
from rule_execution_engine.nodes.registry import NodeRegistry
from rule_execution_engine.spatial.geometry import Object


class VLMConfig(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    model: str = Field(default="gpt-4o-mini")
    prompt: str = Field(
        default=(
            "List every visible object. "
            "Return JSON: {\"objects\": [{\"class_name\": str, \"x\": int, \"y\": int, "
            "\"w\": int, \"h\": int, \"confidence\": float}]}"
        )
    )
    api_key_env: str = Field(default="OPENAI_API_KEY")
    max_tokens: int = Field(default=1024, ge=1)


@NodeRegistry.register("vlm", config_class=VLMConfig)
class VLMNode(BaseNode):
    """Source node: sends input images to a Vision Language Model API.

    Input:  ImageStream   — List[np.ndarray] (one array per frame)
    Output: ObjectStream  — Objects parsed from the structured VLM response

    The prompt instructs the model to return bounding boxes in pixel coordinates.
    The API key is read from an environment variable (default: OPENAI_API_KEY).

    openai package must be installed: pip install openai
    """

    @property
    def input_ports(self) -> List[PortDefinition]:
        return [PortDefinition("input", PortType.ImageStream, "Input image frames")]

    @property
    def output_ports(self) -> List[PortDefinition]:
        return [PortDefinition("output", PortType.ObjectStream, "Objects parsed from VLM response")]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        images: List[np.ndarray] = inputs.get("input", [])
        if not images:
            return {"output": []}

        cfg: VLMConfig = self._parsed_config  # type: ignore[assignment]
        api_key = os.environ.get(cfg.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"API key not found. Set the '{cfg.api_key_env}' environment variable."
            )

        all_objects: List[Object] = []
        for image in images:
            all_objects.extend(self._call_vlm(image, cfg, api_key))
        return {"output": all_objects}

    def _call_vlm(self, image: np.ndarray, cfg: VLMConfig, api_key: str) -> List[Object]:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package is required for VLMNode. "
                "Install with: pip install openai"
            )

        import cv2

        _, buf = cv2.imencode(".png", image)
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=cfg.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": cfg.prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=cfg.max_tokens,
        )

        content = response.choices[0].message.content or "{}"
        return _parse_objects(content)


def _parse_objects(content: str) -> List[Object]:
    """Parse VLM JSON response into Object instances, skipping malformed entries."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return []

    objects = []
    for item in data.get("objects", []):
        try:
            objects.append(Object(
                x=float(item["x"]),
                y=float(item["y"]),
                w=float(item["w"]),
                h=float(item["h"]),
                confidence=float(item.get("confidence", 1.0)),
                class_name=str(item["class_name"]),
            ))
        except (KeyError, ValueError, TypeError):
            continue
    return objects
