#
# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
#

from dataclasses import dataclass


@dataclass
class DirectSolution:
    """Wrapper for a solution produced directly by a repair or synthesis strategy.

    Signals to the runner that the strategy has already produced final code
    which should be validated directly, rather than fed back into
    ``model.generate()`` as a prompt.

    Attributes:
        code: The ready-to-validate solution string.
    """

    code: str
