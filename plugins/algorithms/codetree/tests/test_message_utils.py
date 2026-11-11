# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for codetree.generators.message_utils."""

from codetree.generators.message_utils import Message, message_to_str, messages_to_str


def test_message_dataclass():
    msg = Message(role="system", content="You are a helper.")
    assert msg.role == "system"
    assert msg.content == "You are a helper."


def test_message_to_str():
    msg = Message(role="user", content="Solve x + y")
    assert message_to_str(msg) == "user: Solve x + y"


def test_messages_to_str():
    msgs = [
        Message(role="system", content="System instruction"),
        Message(role="user", content="User question"),
        Message(role="assistant", content="Assistant reply"),
    ]
    formatted = messages_to_str(msgs)
    assert formatted == "system: System instruction\nuser: User question\nassistant: Assistant reply"


def test_messages_to_str_empty_list():
    assert messages_to_str([]) == ""
