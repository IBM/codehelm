# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for codetree.test_connectivity."""

import pytest
from codetree.test_connectivity import check_connectivity, resolve_env_vars


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

    import litellm
    monkeypatch.setattr(litellm, "completion", lambda **kwargs: DummyResponse())

    # Should not raise SystemExit
    check_connectivity(str(config_file))
