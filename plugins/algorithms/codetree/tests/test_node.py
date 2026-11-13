# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for codetree.node (Node data structure)."""

from codetree.node import Node


def test_node_default_initialization():
    node = Node(solution="def foo(): pass")
    assert node.solution == "def foo(): pass"
    assert node.parent is None
    assert node.children == []
    assert node.value == 0
    assert node.depth == 0
    assert node.reflection == ""
    assert node.test_feedback == ""
    assert node.strategy == ""
    assert node.pass_visible is False


def test_node_custom_initialization():
    parent = Node(solution="def parent(): pass", depth=1, strategy="strat1")
    child = Node(
        solution="def child(): pass",
        parent=parent,
        strategy="strat2",
        reflection="reflection text",
        depth=2,
    )

    assert child.solution == "def child(): pass"
    assert child.parent is parent
    assert child.strategy == "strat2"
    assert child.reflection == "reflection text"
    assert child.depth == 2
    assert child.children == []
    assert child.value == 0


def test_node_tree_hierarchy():
    root = Node(solution="root code", depth=1)
    child1 = Node(solution="child 1", parent=root, depth=2)
    child2 = Node(solution="child 2", parent=root, depth=2)
    root.children.extend([child1, child2])

    assert len(root.children) == 2
    assert root.children[0].parent is root
    assert root.children[1].parent is root
    assert root.children[0].depth == 2
    assert root.children[1].depth == 2


def test_node_attribute_mutation():
    node = Node(solution="test")
    node.value = 0.8
    node.test_feedback = "1 of 2 passed"
    node.pass_visible = True
    node.solution = "updated test"

    assert node.value == 0.8
    assert node.test_feedback == "1 of 2 passed"
    assert node.pass_visible is True
    assert node.solution == "updated test"
