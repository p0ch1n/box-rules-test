"""Tests for VLMNode."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rule_execution_engine.nodes.vlm_node import VLMNode, _parse_objects
from rule_execution_engine.schema.models import NodeConfig


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _make_node(
    model: str = "gpt-4o-mini",
    prompt: str = "detect objects",
    api_key_env: str = "OPENAI_API_KEY",
    max_tokens: int = 512,
) -> VLMNode:
    config = NodeConfig(
        id="vlm-1",
        type="vlm",
        position={"x": 0, "y": 0},
        config={
            "model": model,
            "prompt": prompt,
            "api_key_env": api_key_env,
            "max_tokens": max_tokens,
        },
    )
    return VLMNode(config)


def _fake_image() -> np.ndarray:
    return np.zeros((64, 64, 3), dtype=np.uint8)


def _vlm_response(objects: list) -> MagicMock:
    msg = MagicMock()
    msg.content = json.dumps({"objects": objects})
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ------------------------------------------------------------------ #
# Port declarations
# ------------------------------------------------------------------ #


def test_input_port_is_image_stream():
    node = _make_node()
    assert len(node.input_ports) == 1
    assert node.input_ports[0].port_type.value == "ImageStream"


def test_output_port_is_object_stream():
    node = _make_node()
    assert len(node.output_ports) == 1
    assert node.output_ports[0].port_type.value == "ObjectStream"


# ------------------------------------------------------------------ #
# execute() — guard cases
# ------------------------------------------------------------------ #


def test_empty_images_returns_empty_without_api_call(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    node = _make_node()
    result = node.execute({"input": []})
    assert result == {"output": []}


def test_missing_api_key_raises_runtime_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    node = _make_node(api_key_env="OPENAI_API_KEY")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        node.execute({"input": [_fake_image()]})


def test_custom_env_var_name(monkeypatch):
    monkeypatch.delenv("MY_VLM_KEY", raising=False)
    node = _make_node(api_key_env="MY_VLM_KEY")
    with pytest.raises(RuntimeError, match="MY_VLM_KEY"):
        node.execute({"input": [_fake_image()]})


# ------------------------------------------------------------------ #
# execute() — happy path (mocked openai + cv2)
# ------------------------------------------------------------------ #


def _patch_vlm(node: VLMNode, api_key: str, response: MagicMock):
    """Context manager stack: patches openai and cv2.imencode inside _call_vlm."""
    import unittest.mock as mock

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = response

    openai_mod = MagicMock()
    openai_mod.OpenAI.return_value = mock_client

    cv2_mod = MagicMock()
    fake_buf = MagicMock()
    fake_buf.tobytes.return_value = b"\x89PNG"
    cv2_mod.imencode.return_value = (True, fake_buf)

    return (
        mock.patch.dict("sys.modules", {"openai": openai_mod, "cv2": cv2_mod}),
        openai_mod,
        mock_client,
    )


def test_execute_parses_vlm_response_into_objects(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    node = _make_node()

    api_objects = [
        {"class_name": "person", "x": 10, "y": 20, "w": 50, "h": 100, "confidence": 0.9},
        {"class_name": "car", "x": 200, "y": 80, "w": 120, "h": 60, "confidence": 0.75},
    ]
    response = _vlm_response(api_objects)

    import unittest.mock as mock

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = response
    openai_mod = MagicMock()
    openai_mod.OpenAI.return_value = mock_client
    cv2_mod = MagicMock()
    fake_buf = MagicMock()
    fake_buf.tobytes.return_value = b"\x89PNG"
    cv2_mod.imencode.return_value = (True, fake_buf)

    with mock.patch.dict("sys.modules", {"openai": openai_mod, "cv2": cv2_mod}):
        result = node.execute({"input": [_fake_image()]})

    objs = result["output"]
    assert len(objs) == 2
    assert objs[0].class_name == "person"
    assert objs[0].x == 10.0
    assert objs[0].confidence == pytest.approx(0.9)
    assert objs[1].class_name == "car"


def test_execute_flattens_objects_across_frames(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    node = _make_node()

    single_object = [{"class_name": "truck", "x": 0, "y": 0, "w": 10, "h": 10, "confidence": 0.8}]
    response = _vlm_response(single_object)

    import unittest.mock as mock

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = response
    openai_mod = MagicMock()
    openai_mod.OpenAI.return_value = mock_client
    cv2_mod = MagicMock()
    fake_buf = MagicMock()
    fake_buf.tobytes.return_value = b"\x89PNG"
    cv2_mod.imencode.return_value = (True, fake_buf)

    with mock.patch.dict("sys.modules", {"openai": openai_mod, "cv2": cv2_mod}):
        result = node.execute({"input": [_fake_image(), _fake_image()]})

    # Two frames, one object each → two total
    assert len(result["output"]) == 2
    assert mock_client.chat.completions.create.call_count == 2


def test_execute_passes_model_and_prompt_to_api(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    node = _make_node(model="gpt-4o", prompt="find all people")

    import unittest.mock as mock

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _vlm_response([])
    openai_mod = MagicMock()
    openai_mod.OpenAI.return_value = mock_client
    cv2_mod = MagicMock()
    fake_buf = MagicMock()
    fake_buf.tobytes.return_value = b"\x89PNG"
    cv2_mod.imencode.return_value = (True, fake_buf)

    with mock.patch.dict("sys.modules", {"openai": openai_mod, "cv2": cv2_mod}):
        node.execute({"input": [_fake_image()]})

    call_kwargs = mock_client.chat.completions.create.call_args
    assert call_kwargs.kwargs["model"] == "gpt-4o"
    msg_content = call_kwargs.kwargs["messages"][0]["content"]
    text_parts = [p for p in msg_content if p["type"] == "text"]
    assert text_parts[0]["text"] == "find all people"


def test_openai_not_installed_raises_import_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    node = _make_node()

    import unittest.mock as mock
    import sys

    # Remove openai from sys.modules to simulate not installed
    with mock.patch.dict("sys.modules", {"openai": None}):
        with pytest.raises(ImportError, match="pip install openai"):
            node.execute({"input": [_fake_image()]})


# ------------------------------------------------------------------ #
# _parse_objects unit tests
# ------------------------------------------------------------------ #


def test_parse_objects_valid_response():
    content = json.dumps({
        "objects": [
            {"class_name": "person", "x": 5, "y": 10, "w": 30, "h": 60, "confidence": 0.95},
        ]
    })
    objs = _parse_objects(content)
    assert len(objs) == 1
    assert objs[0].class_name == "person"
    assert objs[0].w == 30.0


def test_parse_objects_skips_malformed_entries():
    content = json.dumps({
        "objects": [
            {"class_name": "person", "x": 5, "y": 10, "w": 30, "h": 60},  # valid, no confidence
            {"class_name": "car"},                                            # missing x/y/w/h
            {"x": 0, "y": 0, "w": 10, "h": 10, "confidence": 0.5},         # missing class_name
        ]
    })
    objs = _parse_objects(content)
    # Only first entry is valid; second and third are malformed
    assert len(objs) == 1
    assert objs[0].confidence == pytest.approx(1.0)  # default when absent


def test_parse_objects_invalid_json_returns_empty():
    assert _parse_objects("not json at all") == []


def test_parse_objects_empty_objects_list():
    assert _parse_objects(json.dumps({"objects": []})) == []


def test_parse_objects_missing_objects_key():
    assert _parse_objects(json.dumps({"result": []})) == []
