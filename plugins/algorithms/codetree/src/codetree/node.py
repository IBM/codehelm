# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
# Derived from https://github.com/SalesforceAIResearch/CodeTree (Apache-2.0); modified for CodeHelm.
# Original License below.

# LICENSE HEADER MANAGED BY add-license-header
#
# /*
#  * Copyright (c) 2023, Salesforce, Inc.
#  * SPDX-License-Identifier: Apache-2
#  *
#  * Licensed under the Apache License, Version 2.0 (the "License");
#  * you may not use this file except in compliance with the License.
#  * You may obtain a copy of the License at
#  *
#  *     http://www.apache.org/licenses/LICENSE-2.0
#  *
#  * Unless required by applicable law or agreed to in writing, software
#  * distributed under the License is distributed on an "AS IS" BASIS,
#  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  * See the License for the specific language governing permissions and
#  * limitations under the License.
#  */
#

import math


class Node:
    """Search tree node shared by all search strategies."""

    def __init__(
        self,
        solution: str,
        parent: "Node | None" = None,
        strategy: str = "",
        reflection: str = "",
        depth: int = 0,
    ):
        self.solution = solution
        self.parent = parent
        self.children: list["Node"] = []
        self.value: float = 0
        self.visits: int = 0
        self.context: str = ""
        self.depth = depth
        self.reflection = reflection
        self.test_feedback: str = ""
        self.strategy = strategy
        self.pass_visible: bool = False

    def uct(self, exploration_weight: float = 1.0) -> float:
        if self.visits == 0:
            return self.value
        return (self.value / self.visits) + exploration_weight * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def best_child(self) -> "Node | None":
        if not self.children:
            return None
        return max(self.children, key=lambda child: child.uct())

    def best_child_value(self) -> "Node | None":
        if not self.children:
            return None
        return max(self.children, key=lambda child: child.value)

    def sort_children_by_value(self, reverse: bool = False) -> None:
        self.children.sort(key=lambda x: x.value, reverse=reverse)

    def update(self, reward: float) -> None:
        self.visits += 1
        self.value += reward
