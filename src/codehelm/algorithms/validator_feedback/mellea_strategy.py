"""Mellea sampling strategy for validator feedback-guided generation.

:class:`MelleaFeedbackStrategy` bridges :class:`~codehelm.algorithms.validator_feedback.feeback.Feedback`
into Mellea's :class:`~mellea.stdlib.sampling.BaseSamplingStrategy` protocol so
that validator feedback-guided repair can be used as a drop-in ``strategy`` argument to
``MelleaSession.instruct()``.
"""

from mellea.core import Component, Context
from mellea.stdlib.components import Instruction
from mellea.stdlib.sampling import BaseSamplingStrategy

from codehelm.algorithms.validator_feedback.feeback import Feedback


class MelleaFeedbackStrategy(BaseSamplingStrategy):
    """Mellea :class:`BaseSamplingStrategy` that drives validator feedback repair.

    On each failed attempt Mellea calls :meth:`repair`, which delegates to the
    :class:`~codehelm.algorithms.validator_feedback.feeback.Feedback` instance to compute
    the next retry prompt including failure feedback.

    Args:
        feedback_strategy: A configured :class:`Feedback` instance (defaults to ``Feedback()``).
        loop_budget: Maximum number of repair iterations (passed to
            :class:`BaseSamplingStrategy`).
    """

    _active_instance: "MelleaFeedbackStrategy | None" = None

    def __init__(
        self, feedback_strategy: Feedback | None = None, loop_budget: int = 3
    ) -> None:
        super().__init__(loop_budget=loop_budget)
        self._feedback = feedback_strategy if feedback_strategy is not None else Feedback()
        self._current_prompt: str = ""

    @classmethod
    def set_active(cls, instance: "MelleaFeedbackStrategy | None") -> None:
        cls._active_instance = instance

    @staticmethod
    def repair(
        old_ctx: Context,
        new_ctx: Context,
        past_actions: list[Component],
        past_results,
        past_val,
    ) -> tuple[Component, Context]:
        """Return the feedback-guided prompt as the next generation component.

        Mellea calls this after each failed validation attempt.
        ``past_val`` is ``list[list[tuple[Requirement, ValidationResult]]]``; the
        failure reason is extracted from the first failing ``ValidationResult`` in
        the most recent attempt.
        """
        strategy = MelleaFeedbackStrategy._active_instance
        if strategy is None:
            # Fallback: plain rejection sampling — return original action.
            return past_actions[-1], old_ctx

        # past_val is list[list[tuple[Requirement, ValidationResult]]].
        # Extract the reason from the first failed ValidationResult in the last attempt.
        failure_info = ""
        if past_val:
            last_attempt = past_val[-1]  # list[tuple[Requirement, ValidationResult]]
            for _req, val_result in last_attempt:
                if not bool(val_result):
                    failure_info = val_result.reason or ""
                    break

        repair_prompt: str = strategy._feedback.repair(
            strategy._current_prompt, failure_info
        )

        feedback_component = Instruction(description=repair_prompt)
        return feedback_component, old_ctx

    @staticmethod
    def select_from_failure(sampled_actions, sampled_results, sampled_val) -> int:
        """Return the index of the last attempt as the best-effort fallback."""
        return len(sampled_results) - 1
