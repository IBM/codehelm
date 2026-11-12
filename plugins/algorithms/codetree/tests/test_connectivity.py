# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the connectivity check of a codetree LiteLLM config."""

import os
import sys

import litellm
import pytest
import yaml
from codehelm.models.litellm import LiteLLMModel


def resolve_env_vars(obj):
    """Recursively replace ``"env:VAR_NAME"`` strings (in dicts and lists) with the
    value of the environment variable. Raises ``KeyError`` if the variable is unset."""
    if isinstance(obj, dict):
        return {key: resolve_env_vars(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [resolve_env_vars(value) for value in obj]
    if isinstance(obj, str) and obj.startswith("env:"):
        var_name = obj[4:]
        if var_name not in os.environ:
            raise KeyError(
                f"Config references environment variable '{var_name}' but it is not set."
            )
        return os.environ[var_name]
    return obj


def check_connectivity(config_path):
    """Send one small request to the model described in *config_path*.

    Exits with status 1 if the config is missing, references an unset environment
    variable, or the request fails.
    """
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        model_cfg = resolve_env_vars(config["model"]["model_config"])
    except (OSError, KeyError) as e:
        print(f"Connectivity check failed: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        response = litellm.completion(
            model=LiteLLMModel._resolve_model(
                model_cfg["model_name"], model_cfg.get("api_base")
            ),
            messages=[{"role": "user", "content": "Reply with a short greeting."}],
            max_tokens=16,
            api_base=model_cfg.get("api_base"),
            api_key=model_cfg.get("api_key"),
        )
        print(f"Connectivity OK: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Connectivity check failed: {e}", file=sys.stderr)
        sys.exit(1)


def test_resolve_env_vars_plain_values():
    obj = {"a": 1, "b": "hello", "c": [1, 2, "test"]}
    resolved = resolve_env_vars(obj)
    assert resolved == obj


def test_resolve_env_vars_with_existing_env(monkeypatch):
    monkeypatch.setenv("MY_TEST_KEY", "secret_value_123")
    monkeypatch.setenv("MY_MODEL", "gpt-4")

    config = {
        "api_key": "env:MY_TEST_KEY",
        "nested": {
            "model": "env:MY_MODEL",
            "count": 5,
        },
        "list_items": ["env:MY_MODEL", "plain_text"],
    }

    resolved = resolve_env_vars(config)
    assert resolved["api_key"] == "secret_value_123"
    assert resolved["nested"]["model"] == "gpt-4"
    assert resolved["nested"]["count"] == 5
    assert resolved["list_items"] == ["gpt-4", "plain_text"]


def test_resolve_env_vars_missing_env_raises(monkeypatch):
    monkeypatch.delenv("UNSET_ENV_VAR", raising=False)
    with pytest.raises(KeyError, match="UNSET_ENV_VAR"):
        resolve_env_vars({"key": "env:UNSET_ENV_VAR"})


def test_check_connectivity_missing_config_file(tmp_path):
    missing_path = str(tmp_path / "non_existent.yaml")
    with pytest.raises(SystemExit) as exc_info:
        check_connectivity(missing_path)
    assert exc_info.value.code == 1


def test_check_connectivity_missing_env_var(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("model:\n  model_config:\n    api_key: env:MISSING_API_KEY\n")
    monkeypatch.delenv("MISSING_API_KEY", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        check_connectivity(str(config_file))
    assert exc_info.value.code == 1


def test_check_connectivity_success(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "model:\n"
        "  model_config:\n"
        "    model_name: custom-model\n"
        "    api_base: http://localhost:8000/v1\n"
        "    api_key: dummy-key\n"
    )

    class DummyChoice:
        class Message:
            content = "Connectivity test successful!"
        message = Message()

    class DummyResponse:
        choices = [DummyChoice()]

    monkeypatch.setattr(litellm, "completion", lambda **kwargs: DummyResponse())

    # Should not raise SystemExit
    check_connectivity(str(config_file))


def test_check_connectivity_request_failure_exits(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "model:\n"
        "  model_config:\n"
        "    model_name: custom-model\n"
        "    api_key: dummy-key\n"
    )

    def boom(**kwargs):
        raise ConnectionError("unreachable")

    monkeypatch.setattr(litellm, "completion", boom)

    with pytest.raises(SystemExit) as exc_info:
        check_connectivity(str(config_file))
    assert exc_info.value.code == 1
