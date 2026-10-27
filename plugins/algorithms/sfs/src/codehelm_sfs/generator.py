#
# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
#
# Derived from codespace-optimization/sfs (Apache-2.0); modified for CodeHelm.
#

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from dataclasses import dataclass

from .model_adapter import ModelAdapter
from .types import Message, StrategyReturn

logger = logging.getLogger(__name__)


@dataclass
class GeneratorConfig:
    """Configuration for LLM prompt construction and response parsing."""

    code_block_instruction: str

    parse_code_block: Callable[[str], str]
    add_code_block: Callable[[str], str]
    parse_tests: Callable[[str], list[str]]
    is_syntax_valid: Callable[[str], bool]
    parse_ideas: Callable[[str, int], list[str]]
    parse_insights: Callable[[str], str]

    reflection_chat_instruction: str
    reflection_few_shot: list[Message]
    simple_chat_instruction: str
    simple_generate_few_shot: list[Message]
    self_reflection_chat_instruction: str
    self_reflection_few_shot: list[Message]
    test_generation_chat_instruction: str
    test_generation_few_shot: list[Message]
    test_generation_chat_instruction_cot: str
    test_generation_few_shot_cot: list[Message]
    test_refine_chat_instruction: str
    test_refine_few_shot: list[Message]
    idea_generation_instruction: str
    idea_generation_few_shot: list[Message]
    strategy_generation_instruction: str
    strategy_generation_few_shot: list[Message]
    summary_update_instruction: str
    summary_update_few_shot: list[Message]
    scout_idea_generation_instruction: str
    scout_idea_generation_few_shot: list[Message]

    test_generation_chat_instruction_strict: str = ""
    test_generation_few_shot_strict: list[Message] | None = None


class GenericChatGenerator:
    """Domain-agnostic LLM generator driving seed synthesis, test generation, and MCTS mutations."""

    def __init__(self, config: GeneratorConfig):
        self.config = config

    def generate_self_reflection(
        self, func: str, feedback: str, model: ModelAdapter, temperature: float
    ) -> str:
        """Generate a self-reflection on why a function failed tests."""
        if not model.is_chat:
            raise ValueError("Model is not chat based")

        if self.config.self_reflection_few_shot is not None:
            messages = (
                [
                    Message(
                        role="system",
                        content=self.config.self_reflection_chat_instruction,
                    )
                ]
                + self.config.self_reflection_few_shot
                + [
                    Message(
                        role="user",
                        content=f"[function impl]:\n{self.config.add_code_block(func)}\n\n[unit test results]:\n{feedback}\n\n[self-reflection]:",
                    )
                ]
            )
        else:
            messages = [
                Message(
                    role="system", content=self.config.self_reflection_chat_instruction
                ),
                Message(
                    role="user",
                    content=f"[function impl]:\n{self.config.add_code_block(func)}\n\n[unit test results]:\n{feedback}\n\n[self-reflection]:",
                ),
            ]
        reflection = model.generate(input_data=messages, temperature=temperature)[0]
        return reflection.content

    def strategist_generate_ideas(
        self,
        func_sig: str,
        prev_func_impl: str,
        num_ideas: int,
        model: ModelAdapter,
        strategy: str,
        feedback: str,
        self_reflection: str,
        temperature: float,
        num_comps: int = 1,
    ) -> list[str]:
        """Generate candidate improvement ideas without summary context."""
        if strategy != "reflexion":
            raise ValueError(
                f"Invalid strategy: given `{strategy}` but expected `reflexion`"
            )

        prompt = f"{self.config.idea_generation_instruction}"
        messages = (
            [Message(role="system", content=prompt)]
            + self.config.idea_generation_few_shot
            + [
                Message(
                    role="user",
                    content=f"[prompt]:\n{func_sig}\n\n[previous impl]:{self.config.add_code_block(prev_func_impl)}\n\n[unit test results from previous impl]:\n{feedback}\n\n What are some reflections you can draw and what are {num_ideas} ways in which you can fix the previous impl so that it passes all the tests? Be as specific and concrete as possible, mentioning specific code blocks or helper functions that you will modify. Remember to number your ideas as Idea 1:, Idea 2:, ... etc. \n\n[reflection and {num_ideas} improvement ideas]:",
                )
            ]
        )
        responses = model.generate(
            input_data=messages, num_comps=num_comps, temperature=temperature
        )
        return self.config.parse_ideas(responses[0].content, num_ideas)

    def generate_ideas_with_summary(
        self,
        func_sig: str,
        prev_func_impl: str,
        num_ideas: int,
        model: ModelAdapter,
        feedback: str,
        temperature: float,
        summary: str,
        num_comps: int = 1,
        is_few_shot: bool = True,
    ) -> list[str]:
        """Scout step: generate improvement ideas incorporating running insight summary."""
        sys_message = Message(
            role="system", content=self.config.scout_idea_generation_instruction
        )
        user_message = Message(
            role="user",
            content=f"[prompt]:\n{func_sig}\n\n[previous impl]:{self.config.add_code_block(prev_func_impl)}\n\n[unit test results from previous impl]:\n{feedback}\n\n[general insights]:\n{summary}\n\nWhat are some reflections you can draw based on the results and insights, and what are {num_ideas} different ways in which you can fix the previous implementation so that it passes all the tests? Be as specific and concrete as possible, mentioning specific code blocks or helper functions that you will modify. Remember to number your ideas as Idea 1:, Idea 2:, etc. \n\n[reflection and {num_ideas} improvement ideas]:",
        )
        if is_few_shot:
            messages = (
                [sys_message]
                + self.config.scout_idea_generation_few_shot
                + [user_message]
            )
        else:
            messages = [sys_message, user_message]

        responses = model.generate(
            input_data=messages, num_comps=num_comps, temperature=temperature
        )
        return self.config.parse_ideas(responses[0].content, num_ideas)

    def strategist_generate_strategy(
        self,
        func_sig: str,
        model: ModelAdapter,
        strategy: str,
        prev_func_impl: str,
        feedback: str,
        self_reflection: str,
        idea_to_implement: str,
        temperature: float,
        num_comps: int = 1,
    ) -> StrategyReturn:
        """Strategist step: implement code modification based on selected idea."""
        if strategy == "reflexion":
            prompt = f"{self.config.strategy_generation_instruction}\n{self.config.code_block_instruction}"
            messages = (
                [Message(role="system", content=prompt)]
                + self.config.strategy_generation_few_shot
                + [
                    Message(
                        role="user",
                        content=f"[prompt]:\n{func_sig}\n\n[previous impl]: {self.config.add_code_block(prev_func_impl)}\n\n[unit test results from previous impl]:\n{feedback} \n\n[reflection]:\n{self_reflection} \n\n[improvement idea]:\n{idea_to_implement} \nIncorporate the idea above into the previous impl as best as you can. \n\n[improved impl]:",
                    )
                ]
            )
        elif strategy == "outcome":
            prompt = f"{self.config.strategy_generation_instruction}\n{self.config.code_block_instruction}"
            messages = (
                [Message(role="system", content=prompt)]
                + self.config.strategy_generation_few_shot
                + [
                    Message(
                        role="user",
                        content=f"[prompt]:\n{func_sig}\n\n[previous impl]: {self.config.add_code_block(prev_func_impl)}\n\n[unit test results from previous impl]:\n{feedback} \n\n[improvement idea]:\n{idea_to_implement} \nIncorporate the idea above into the previous impl as best as you can. \n\n[improved impl]:",
                    )
                ]
            )
        else:
            raise ValueError(f"Invalid strategy: `{strategy}`")

        responses = model.generate(
            input_data=messages, num_comps=num_comps, temperature=temperature
        )
        parsed_code = self.config.parse_code_block(responses[0].content) or ""
        return StrategyReturn(
            code=parsed_code,
            logprob=responses[0].logprob,
        )

    def generate_internal_tests(
        self,
        func_sig: str,
        max_num_tests: int,
        model: ModelAdapter,
        entry_point: str,
        temperature: float,
        method: str = "simple",
    ) -> list[str]:
        """Generate visible validation tests for a coding problem."""
        if method == "simple":
            messages = (
                [
                    Message(
                        role="system",
                        content=self.config.test_generation_chat_instruction,
                    )
                ]
                + self.config.test_generation_few_shot
                + [
                    Message(
                        role="user",
                        content=f"[object name]:\n{entry_point}\n\n[object description]:\n{func_sig}\n\n[{max_num_tests} unit tests]:",
                    )
                ]
            )
        elif method == "strict":
            instruction = (
                self.config.test_generation_chat_instruction_strict
                or self.config.test_generation_chat_instruction
            )
            few_shot = (
                self.config.test_generation_few_shot_strict
                or self.config.test_generation_few_shot
            )
            messages = (
                [Message(role="system", content=instruction)]
                + few_shot
                + [
                    Message(
                        role="user",
                        content=f"[object name]:\n{entry_point}\n\n[object description]:\n{func_sig}\n\n[{max_num_tests} unit tests]:",
                    )
                ]
            )
        else:
            raise ValueError(f"Invalid test generation method: `{method}`")

        outputs = model.generate(
            input_data=messages, max_tokens=1024, temperature=temperature
        )
        all_tests = self.config.parse_tests(outputs[0].content)
        valid_tests = [test for test in all_tests if self.config.is_syntax_valid(test)]
        return valid_tests[:max_num_tests]

    def generate_func_impl(
        self,
        func_sig: str,
        strategy: str,
        model: ModelAdapter,
        entry_point: str,
        temperature: float,
        num_comps: int = 1,
        preamble: str = "",
        is_few_shot: bool = True,
        prev_func_impl: str | None = None,
        feedback: str | None = None,
        self_reflection: str | None = None,
    ) -> list[StrategyReturn]:
        """Generate seed solution implementations."""
        if strategy != "reflexion" and strategy != "simple":
            raise ValueError(
                f"Invalid strategy: given `{strategy}` but expected `reflexion` or `simple`"
            )

        if strategy == "reflexion":
            if prev_func_impl is None or feedback is None or self_reflection is None:
                raise ValueError(
                    "reflexion strategy requires prev_func_impl, feedback, self_reflection"
                )
            system_prompt = f"{self.config.reflection_chat_instruction}\n{self.config.code_block_instruction}\n{preamble}"
            messages = (
                [Message(role="system", content=system_prompt)]
                + self.config.reflection_few_shot
                + [
                    Message(
                        role="user",
                        content=f"[prompt]:\n{func_sig}\n\n[previous impl]:\n{self.config.add_code_block(prev_func_impl or '')}\n\n[unit test results from previous impl]:\n{feedback}\n\n[reflection on previous impl]:\n{self_reflection}\n\n[improved impl]:",
                    )
                ]
            )
        else:
            system_prompt = f"{self.config.simple_chat_instruction}\n{self.config.code_block_instruction}\n{preamble}"
            if is_few_shot:
                messages = (
                    [Message(role="system", content=system_prompt)]
                    + self.config.simple_generate_few_shot
                    + [
                        Message(
                            role="user",
                            content=f"[object name]:\n{entry_point}\n\n[object description]:\n{func_sig}\n\n[object impl]:",
                        )
                    ]
                )
            else:
                messages = [
                    Message(role="system", content=system_prompt),
                    Message(role="user", content=func_sig),
                ]

        func_bodies = model.generate(
            input_data=messages, num_comps=num_comps, temperature=temperature
        )
        return [
            StrategyReturn(
                code=self.config.parse_code_block(body.content) or "",
                logprob=body.logprob,
            )
            for body in func_bodies
        ]

    def generate_updated_summary(
        self,
        prompt: str,
        prev_impl: str,
        prev_feedback: str,
        improvement_idea: str,
        new_impl: str,
        model: ModelAdapter,
        new_feedback: str,
        temperature: float,
        summary: str,
        score_difference: float,
        num_comps: int = 1,
        is_few_shot: bool = True,
    ) -> str:
        """Update insight summary based on reflection."""
        sys_message = Message(
            role="system", content=self.config.summary_update_instruction
        )
        user_message = Message(
            role="user",
            content=f"[prompt]:\n{prompt}\n\n[previous implementation]:\n{self.config.add_code_block(prev_impl)}\n\n[previous implementation feedback]:\n{prev_feedback}\n\n[improvement idea]:\n{improvement_idea}\n\n[new implementation]:\n{self.config.add_code_block(new_impl)}\n\n[new implementation feedback]:\n{new_feedback}\n\n[score improvement]:\n{score_difference}\n\n[previous insights]:\n{summary}\n\n# Reflect on how the improvement idea changed the performance of the code implementation and update your insights accordingly.\n[updated insights]:",
        )
        if is_few_shot:
            messages = (
                [sys_message] + self.config.summary_update_few_shot + [user_message]
            )
        else:
            messages = [sys_message, user_message]

        responses = model.generate(
            input_data=messages, num_comps=num_comps, temperature=temperature
        )
        return self.config.parse_insights(responses[0].content)

    @staticmethod
    def sample_n_random(items: list[str], n: int) -> list[str]:
        if n <= 0:
            return []
        if n >= len(items):
            return items
        return random.sample(items, n)
