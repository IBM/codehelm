#
# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
#
# Derived from codespace-optimization/sfs (Apache-2.0); modified for CodeHelm.
#

"""Qiskit-flavoured generator configuration and prompts.

Configures GenericChatGenerator with system prompts and few-shot examples
specialized for IBM Qiskit quantum-computing code.
"""

from __future__ import annotations

from .generator import GeneratorConfig, GenericChatGenerator
from .model_adapter import ModelAdapter
from .parse import (
    add_code_block,
    parse_code_block,
    parse_improvement_ideas,
    parse_tests,
    parse_updated_insights,
    py_is_syntax_valid,
)
from .types import Message

USE_QISKIT_CODEBLOCK_INSTRUCTION = (
    "Use a Python code block to write your response. For example:\n"
    "```python\n"
    "from qiskit import QuantumCircuit\n"
    "qc = QuantumCircuit(2)\n"
    "qc.h(0)\n"
    "qc.cx(0, 1)\n"
    "```"
)

QISKIT_SIMPLE_CHAT_INSTRUCTION = (
    "You are an AI that only responds with Python/Qiskit code, NOT ENGLISH. "
    "You will be given the name of a Qiskit function, class, or quantum circuit "
    "that you need to implement. "
    "Rules:\n"
    "1. Write your full implementation using the Qiskit SDK, including all necessary "
    "imports\n"
    "2. If multiple functions are described, provide the full implementation"
    " for ALL of them in the same code block.\n"
    "3. Use only standard ASCII in comments, docstrings, and code (e.g. use"
    " |phi^+>, |psi^->, |0>, |1> instead of Unicode |φ⁺⟩, |Ψ⁻⟩, ⁺, ⁻, ⟨, ⟩) to"
    " ensure valid Python syntax.\n"
    "4. Wrap your entire code inside a single ```python ... ``` code block with"
    " no commentary outside."
)

QISKIT_SIMPLE_GENERATE_FEW_SHOT = [
    Message(
        role="user",
        content="""[object name]:
bell_state

[object description]:
Create and return a 2-qubit Bell state (maximally entangled) QuantumCircuit.

[object impl]:""",
    ),
    Message(
        role="assistant",
        content="""```python
from qiskit import QuantumCircuit

def bell_state() -> QuantumCircuit:
    \"\"\"
    Create and return a 2-qubit Bell state (maximally entangled) QuantumCircuit.
    Applies H on qubit 0 then CNOT(0, 1).
    \"\"\"
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc
```""",
    ),
]

QISKIT_REFLECTION_CHAT_INSTRUCTION = (
    "You are an AI Qiskit assistant. You will be given your past"
    " implementation, a series of unit tests, and a hint.\n"
    "Rules:\n"
    "1. Write your full improved Qiskit implementation, including all necessary"
    " imports.\n"
    "2. If the problem contains multiple functions, make sure to include ALL of"
    " them in your final code block.\n"
    "3. Use only standard ASCII notation in comments and docstrings (e.g. use"
    " |phi^+>, |psi^->, |0>, |1> rather than Unicode |φ⁺⟩, |Ψ⁻⟩, ⁺, ⁻, ⟨, ⟩).\n"
    "4. Wrap your implementation in a single ```python ... ``` block with no"
    " trailing text outside the block."
)

QISKIT_REFLEXION_FEW_SHOT = [
    Message(
        role="user",
        content="""[prompt]:
Write a function that creates a QuantumCircuit that prepares the GHZ state on n qubits.

[previous impl]:
```python
from qiskit import QuantumCircuit

def ghz_state(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.h(0)
    # missing CNOT chain
    return qc
```

[unit test results from previous impl]:
Tests passed:

Tests failed:
assert ghz_state(3).num_qubits == 3  # output: 3  (passes)
# statevector check failed: expected equal superposition of |000> and |111>

[reflection on previous impl]:
The implementation applies H to qubit 0 but does not entangle the remaining qubits.
GHZ requires a CNOT chain from qubit 0 to each subsequent qubit so that the state
becomes (|00...0> + |11...1>) / sqrt(2).

[improved impl]:
""",
    ),
    Message(
        role="assistant",
        content="""```python
from qiskit import QuantumCircuit

def ghz_state(n: int) -> QuantumCircuit:
    \"\"\"
    Prepare the n-qubit GHZ state: (|00...0> + |11...1>) / sqrt(2).
    \"\"\"
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(0, i + 1)
    return qc
```""",
    ),
]

QISKIT_SELF_REFLECTION_CHAT_INSTRUCTION = (
    "You are a Qiskit programming assistant. "
    "You will be given a Qiskit circuit or function implementation and a series of "
    "unit tests. Your goal is to write a few sentences explaining why the implementation "
    "is wrong as indicated by the tests. You will need this as a hint when you try again "
    "later. Only provide the short explanation, not the corrected implementation."
)

QISKIT_SELF_REFLECTION_FEW_SHOT = [
    Message(
        role="user",
        content="""[prompt]:
Write a function that applies a Quantum Fourier Transform (QFT) to the first n qubits
of a QuantumCircuit and returns the circuit.

[function impl]:
```python
from qiskit import QuantumCircuit
import numpy as np

def qft(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    for j in range(n):
        qc.h(j)
        for k in range(j + 1, n):
            qc.cp(np.pi / (2 ** (k - j)), j, k)
    return qc
```

[unit test results]:
Tests passing:
assert qft(1).num_qubits == 1

Tests failing:
# Unitary of qft(2) does not match reference QFT matrix — SWAP step is missing.

[self-reflection]:
""",
    ),
    Message(
        role="assistant",
        content=(
            "The implementation applies the correct H gates and controlled-phase rotations "
            "but omits the bit-reversal SWAP step at the end of the QFT. Without this step "
            "the output qubit ordering is reversed relative to the standard QFT definition, "
            "causing the unitary to differ from the reference. Adding swap gates between "
            "qubit i and qubit n-1-i for i in range(n // 2) will fix the output ordering."
        ),
    ),
    Message(
        role="user",
        content="""[prompt]:
Write a function that measures all qubits in a QuantumCircuit in the Z basis
and returns the circuit with classical bits added.

[function impl]:
```python
from qiskit import QuantumCircuit

def measure_all(qc: QuantumCircuit) -> QuantumCircuit:
    qc.measure_all()
    return qc
```

[unit test results]:
Tests passing:
assert measure_all(QuantumCircuit(2)).num_clbits == 2

Tests failing:
assert measure_all(QuantumCircuit(2)).num_qubits == 2  # output: AttributeError

[self-reflection]:
""",
    ),
    Message(
        role="assistant",
        content=(
            "The function calls qc.measure_all() which adds an ancilla classical register "
            "named 'meas' and a BARRIER before measuring; this mutates the circuit in place "
            "and returns None, not the circuit. Returning the mutated circuit directly works, "
            "but the barrier sometimes causes issues with downstream transpilation. "
            "Using qc.measure_all(inplace=True) and then returning qc, or explicitly adding "
            "a ClassicalRegister and calling qc.measure(range(n), range(n)), avoids the None "
            "return issue."
        ),
    ),
]

QISKIT_TEST_GENERATION_CHAT_INSTRUCTION = """You are an AI coding assistant that can write unique, diverse, and comprehensive unit tests for Qiskit quantum-computing objects given their description.

CRITICAL RULES:
1. Each assert statement must be COMPLETELY SELF-CONTAINED and EXECUTABLE on its own.
2. Do NOT split multi-line test setups — keep setup and assertion together on ONE line or use semicolons.
3. Do NOT reference undefined variables.
4. Import Qiskit inside each assert block if needed, or use a semicolon-separated setup line.

For simple circuit-property checks:
```python
assert bell_state().num_qubits == 2, "Bell state has 2 qubits"
assert bell_state().depth() > 0, "Circuit has gates"
```

For statevector / unitary checks use inline setup:
```python
from qiskit.quantum_info import Statevector; sv = Statevector(bell_state()); assert abs(sv[0]) > 0, "amplitude of |00>"
```

DO NOT use pytest or unittest. Keep tests simple and independently executable."""

QISKIT_TEST_GENERATION_FEW_SHOT = [
    Message(
        role="user",
        content="""[object name]:
bell_state

[object description]:
def bell_state() -> QuantumCircuit:
    \"\"\"Create and return a 2-qubit maximally entangled Bell state circuit.\"\"\"

[unit tests]:
""",
    ),
    Message(
        role="assistant",
        content="""```python
assert bell_state().num_qubits == 2, "Bell state must have exactly 2 qubits"
assert bell_state().num_clbits == 0, "No classical bits before measurement"
assert bell_state().depth() >= 2, "Circuit needs at least H + CNOT"
from qiskit.quantum_info import Statevector; sv = Statevector(bell_state()); assert round(abs(sv[0b00])**2, 6) == 0.5, "|00> amplitude squared is 0.5"
from qiskit.quantum_info import Statevector; sv = Statevector(bell_state()); assert round(abs(sv[0b11])**2, 6) == 0.5, "|11> amplitude squared is 0.5"
```""",
    ),
]

QISKIT_TEST_GENERATION_CHAT_INSTRUCTION_STRICT = """You are a unit-test writer for Qiskit quantum-computing functions. Your ONLY job is to output bare assert statements — nothing else.

ABSOLUTE RULES:
1. Output EXACTLY the number of assert statements requested — no more, no fewer.
2. Every line must start with `assert`. No imports at module level, no def, no comments, no blank lines, no prose.
3. Every assert is a single self-contained executable line. Do NOT split across lines.
4. Do NOT wrap asserts in a function or class.
5. If Qiskit imports are needed, inline them: `from qiskit.quantum_info import Statevector; sv = Statevector(f()); assert ...`
6. Compute every expected value yourself — do NOT assert something you are not certain about.
7. Cover meaningful quantum scenarios: qubit/gate counts, statevector amplitudes, circuit depth, transpilation properties.

OUTPUT FORMAT:
```python
assert f(args).property == expected, "description"
```"""

QISKIT_TEST_GENERATION_FEW_SHOT_STRICT = [
    Message(
        role="user",
        content="""[object name]:
ghz_state

[object description]:
def ghz_state(n: int) -> QuantumCircuit:
    \"\"\"Prepare the n-qubit GHZ state (|00...0> + |11...1>) / sqrt(2).\"\"\"

[3 unit tests]:
""",
    ),
    Message(
        role="assistant",
        content="""```python
assert ghz_state(3).num_qubits == 3, "3-qubit GHZ has 3 qubits"
assert ghz_state(1).num_qubits == 1, "1-qubit GHZ is just |+>"
from qiskit.quantum_info import Statevector; sv = Statevector(ghz_state(3)); assert round(abs(sv[0b000])**2, 6) == 0.5, "|000> amplitude"
```""",
    ),
]

QISKIT_STRATEGIST_IDEA_GENERATION_INSTRUCTION = """You are a Qiskit programming assistant. You will be given a Qiskit circuit/function implementation and unit test results. Explain why the implementation failed the tests and propose a few concrete ways to fix it so that it meets the requirements and passes the tests. If you think the unit tests are incorrect, note that too. Only provide the improvement ideas that *modify the implementation*, not the modified code itself.

NOTE: Express quantum states using ASCII notation (e.g. |0>, |1>, |+>, |->, |phi^+>, |psi^->) to avoid Unicode syntax issues.
IMPORTANT: Format your response exactly as follows:
1. Start with a "Reflection:" section explaining why the implementation failed.
2. Then list your improvement ideas using this exact format:
   Idea 1: <your first improvement idea>
   Idea 2: <your second improvement idea>
   Idea 3: <your third improvement idea>

Do NOT use markdown formatting like "**Idea 1:**" — use the exact format "Idea X:" where X is the number.

-----"""

QISKIT_STRATEGIST_IDEA_GENERATION_FEW_SHOT = [
    Message(
        role="user",
        content="""[prompt]:
def qft(n: int) -> QuantumCircuit:
    '''Prepare the n-qubit Quantum Fourier Transform circuit.'''

[previous impl]:
```python
from qiskit import QuantumCircuit
import numpy as np

def qft(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    for j in range(n):
        qc.h(j)
        for k in range(j + 1, n):
            qc.cp(np.pi / float(2 ** (k - j)), j, k)
    return qc
```

[unit test results for previous impl]:
Tests passed:
assert qft(1).num_qubits == 1

Tests failed:
# Unitary does not match reference QFT — SWAP reversal step is missing.

What are some reflections you can draw and what are 2 ways in which you can fix the previous impl so that it passes all the tests? Be as specific and concrete as possible.
[reflection and 2 improvement ideas]:
""",
    ),
    Message(
        role="assistant",
        content="""Reflection: The implementation correctly applies the H gates and controlled-phase rotations but omits the final bit-reversal SWAP step. Without the SWAPs the output qubit ordering is reversed, so the unitary matrix differs from the reference QFT.

Idea 1: Add a SWAP reversal loop after the main QFT body: for i in range(n // 2): qc.swap(i, n - 1 - i). This reorders the output qubits to match the standard QFT definition.
Idea 2: Use Qiskit's built-in QFT from qiskit.circuit.library import QFT and return QFT(n).decompose(), which already includes the correct SWAP step.
""",
    ),
]

QISKIT_SCOUT_IDEA_GENERATION_INSTRUCTION = """You are a Qiskit programming assistant. You will be given a Qiskit circuit/function implementation, unit test results, and general insights from previous attempts. Explain why the implementation failed the tests and propose a few concrete ways to fix it. If you think the unit tests are incorrect, note that too. Only provide the improvement ideas, not the modified code.

IMPORTANT: Format your response exactly as follows:
1. Start with a "Reflection:" section explaining why the implementation failed.
2. Then list your improvement ideas using this exact format:
   Idea 1: <your first improvement idea>
   Idea 2: <your second improvement idea>
   Idea 3: <your third improvement idea>

Do NOT use markdown formatting like "**Idea 1:**" — use the exact format "Idea X:" where X is the number.

-----"""

QISKIT_SCOUT_IDEA_GENERATION_FEW_SHOT = [
    Message(
        role="user",
        content="""[prompt]:
def qft(n: int) -> QuantumCircuit:
    '''Prepare the n-qubit Quantum Fourier Transform circuit.'''

[previous impl]:
```python
from qiskit import QuantumCircuit
import numpy as np

def qft(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    for j in range(n):
        qc.h(j)
        for k in range(j + 1, n):
            qc.cp(np.pi / float(2 ** (k - j)), j, k)
    return qc
```

[unit test results for previous impl]:
Tests passed:
assert qft(1).num_qubits == 1

Tests failed:
# Unitary does not match reference QFT — SWAP reversal step is missing.

[general insights]:
- The QFT requires a bit-reversal permutation (SWAP chain) at the end.

What are some reflections you can draw and what are 2 ways in which you can fix the previous impl so that it passes all the tests?
[reflection and 2 improvement ideas]:
""",
    ),
    Message(
        role="assistant",
        content="""Reflection: The implementation correctly applies the H gates and controlled-phase rotations but omits the final bit-reversal SWAP step (observation). Without the SWAPs the qubit ordering is reversed, causing the unitary to differ from the reference (cause). The general insight confirms this: the bit-reversal permutation is required (insights applied).

Idea 1: Add a SWAP reversal loop after the main body: for i in range(n // 2): qc.swap(i, n - 1 - i).
Idea 2: Replace the manual implementation with qiskit.circuit.library.QFT(n) which already includes the SWAP step.
""",
    ),
]

QISKIT_STRATEGIST_GENERATE_STRATEGY_INSTRUCTION = (
    "You are an AI Qiskit assistant. You will be given your past Qiskit circuit/function "
    "implementation, unit test results, and a specific improvement idea. "
    "Your job is to apply the improvement idea to the previous implementation and return "
    "the corrected Qiskit code."
)

QISKIT_STRATEGIST_GENERATE_STRATEGY_FEW_SHOT = [
    Message(
        role="user",
        content="""[prompt]:
def ghz_state(n: int) -> QuantumCircuit:
    '''Prepare the n-qubit GHZ state.'''

[previous impl]:
```python
from qiskit import QuantumCircuit

def ghz_state(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.h(0)
    return qc
```

[unit test results from previous impl]:
Tests passed:

Tests failed:
assert ghz_state(3).num_qubits == 3
# statevector |000> + |111> check failed

[improvement idea]:
Add a CNOT chain from qubit 0 to each qubit 1..n-1 after the H gate.

Incorporate the idea above into the previous impl as best as you can.

[improved impl]:
""",
    ),
    Message(
        role="assistant",
        content="""
```python
from qiskit import QuantumCircuit

def ghz_state(n: int) -> QuantumCircuit:
    \"\"\"Prepare the n-qubit GHZ state: (|00...0> + |11...1>) / sqrt(2).\"\"\"
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(0, i + 1)
    return qc
```
""",
    ),
]

QISKIT_SUMMARY_UPDATE_INSTRUCTION = (
    "You are a quantum-software programming assistant. You will be given a past Qiskit "
    "implementation, feedback on its performance, an idea for changing the implementation, "
    "a new implementation, feedback on the new implementation, and insights you previously "
    "learned. Update your previous insights based on what you learned from applying the new "
    "idea. The updated insights should be general, focused on why the improvement did or did "
    "not work, and not specific to that particular circuit. Your insights will be used when "
    "suggesting future improvement ideas."
)

QISKIT_SUMMARY_UPDATE_FEW_SHOT = [
    Message(
        role="user",
        content="""[prompt]:
Write a Qiskit function to apply a QFT on n qubits and return the circuit.

[previous implementation]:
```python
from qiskit import QuantumCircuit
import numpy as np

def qft(n):
    qc = QuantumCircuit(n)
    for j in range(n):
        qc.h(j)
        for k in range(j + 1, n):
            qc.cp(np.pi / 2 ** (k - j), j, k)
    return qc
```

[previous implementation feedback]:
Tests passed: num_qubits check
Tests failed: unitary does not match reference QFT

[improvement idea]:
Add SWAP reversal loop: for i in range(n // 2): qc.swap(i, n-1-i)

[new implementation]:
```python
from qiskit import QuantumCircuit
import numpy as np

def qft(n):
    qc = QuantumCircuit(n)
    for j in range(n):
        qc.h(j)
        for k in range(j + 1, n):
            qc.cp(np.pi / 2 ** (k - j), j, k)
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    return qc
```

[new implementation feedback]:
Tests passed: all

[score improvement]:
1.0

[previous insights]:
- The QFT gate sequence must end with a bit-reversal permutation.

Reflect on how the improvement idea changed the performance and update your insights.
[updated insights]:
""",
    ),
    Message(
        role="assistant",
        content="""
*Reflections*
Adding the SWAP reversal loop fixed all test cases, yielding a full score improvement (observation). The standard QFT definition requires the output qubit order to be reversed relative to the order in which rotations are applied; omitting this step produces an incorrect unitary (cause). The improvement idea was precisely targeted and correct (conclusion).

*Updated Insights*
When implementing QFT manually in Qiskit always include the bit-reversal SWAP chain at the end.
When a circuit passes structural checks (qubit count, gate count) but fails unitary checks, look for missing permutation or phase steps rather than incorrect gate parameters.
""",
    ),
]

QISKIT_COT_TEST_GENERATION_CHAT_INSTRUCTION = (
    "You are a quantum software engineer capable of writing unique, diverse, and "
    "comprehensive unit tests for Qiskit objects given their description. Work through "
    "the circuit step by step from input to output for each unit test to ensure correctness. "
    "Use small circuits you can easily verify. "
    "Format: ```python\nassert ...\n```"
)

QISKIT_COT_TEST_GENERATION_FEW_SHOT = [
    Message(
        role="user",
        content="""[object name]:
bell_state

[object description]:
def bell_state() -> QuantumCircuit:
    '''Create a 2-qubit Bell state circuit.'''

[2 unit tests]:
""",
    ),
    Message(
        role="assistant",
        content="""*Test case 1* — structural check
Input: bell_state()
A Bell circuit needs exactly 2 qubits and no classical bits before measurement.
Output: num_qubits == 2
```python
assert bell_state().num_qubits == 2, "Bell state has 2 qubits"
```

*Test case 2* — statevector amplitude check
Input: Statevector(bell_state())
H on qubit 0 creates |+0>, then CNOT entangles to (|00>+|11>)/sqrt(2).
|00> amplitude squared = 0.5
```python
from qiskit.quantum_info import Statevector; sv = Statevector(bell_state()); assert round(abs(sv[0])**2, 6) == 0.5, "|00> amplitude"
```
""",
    ),
]

QISKIT_TEST_REFINE_CHAT_INSTRUCTION = (
    "You are an AI coding assistant that can analyse whether a given Qiskit unit test "
    "is correct and improve it. Provide both the analysis of the original test and the "
    "improved test."
)

QISKIT_TEST_REFINE_FEW_SHOT = [
    Message(
        role="user",
        content="""[object name]:
bell_state

[object description]:
def bell_state() -> QuantumCircuit:
    '''Create a 2-qubit Bell state circuit.'''

[proposed test]:
assert bell_state().depth() == 1, "Bell circuit depth"

[analysis and improved test]:
""",
    ),
    Message(
        role="assistant",
        content="""*Analysis*:
A Bell circuit requires H on qubit 0 (depth 1) followed by CNOT(0,1) (depth 2), so the
minimum depth is 2, not 1. The proposed test would always fail.

*Improved test*:
```python
assert bell_state().depth() >= 2, "Bell circuit needs at least H + CNOT layers"
```""",
    ),
]

qiskit_config = GeneratorConfig(
    code_block_instruction=USE_QISKIT_CODEBLOCK_INSTRUCTION,
    parse_code_block=lambda x: parse_code_block(x, "python") or "",
    add_code_block=lambda x: add_code_block(x, "python"),
    parse_ideas=parse_improvement_ideas,
    parse_insights=parse_updated_insights,
    parse_tests=parse_tests,
    is_syntax_valid=py_is_syntax_valid,
    reflection_chat_instruction=QISKIT_REFLECTION_CHAT_INSTRUCTION,
    simple_chat_instruction=QISKIT_SIMPLE_CHAT_INSTRUCTION,
    self_reflection_chat_instruction=QISKIT_SELF_REFLECTION_CHAT_INSTRUCTION,
    test_generation_chat_instruction=QISKIT_TEST_GENERATION_CHAT_INSTRUCTION,
    test_generation_chat_instruction_strict=QISKIT_TEST_GENERATION_CHAT_INSTRUCTION_STRICT,
    test_generation_chat_instruction_cot=QISKIT_COT_TEST_GENERATION_CHAT_INSTRUCTION,
    test_refine_chat_instruction=QISKIT_TEST_REFINE_CHAT_INSTRUCTION,
    idea_generation_instruction=QISKIT_STRATEGIST_IDEA_GENERATION_INSTRUCTION,
    strategy_generation_instruction=QISKIT_STRATEGIST_GENERATE_STRATEGY_INSTRUCTION,
    summary_update_instruction=QISKIT_SUMMARY_UPDATE_INSTRUCTION,
    scout_idea_generation_instruction=QISKIT_SCOUT_IDEA_GENERATION_INSTRUCTION,
    reflection_few_shot=QISKIT_REFLEXION_FEW_SHOT,
    simple_generate_few_shot=QISKIT_SIMPLE_GENERATE_FEW_SHOT,
    self_reflection_few_shot=QISKIT_SELF_REFLECTION_FEW_SHOT,
    test_generation_few_shot=QISKIT_TEST_GENERATION_FEW_SHOT,
    test_generation_few_shot_strict=QISKIT_TEST_GENERATION_FEW_SHOT_STRICT,
    test_generation_few_shot_cot=QISKIT_COT_TEST_GENERATION_FEW_SHOT,
    test_refine_few_shot=QISKIT_TEST_REFINE_FEW_SHOT,
    idea_generation_few_shot=QISKIT_STRATEGIST_IDEA_GENERATION_FEW_SHOT,
    strategy_generation_few_shot=QISKIT_STRATEGIST_GENERATE_STRATEGY_FEW_SHOT,
    summary_update_few_shot=QISKIT_SUMMARY_UPDATE_FEW_SHOT,
    scout_idea_generation_few_shot=QISKIT_SCOUT_IDEA_GENERATION_FEW_SHOT,
)


class QiskitGenerator(GenericChatGenerator):
    """Generator configured with Qiskit-domain system prompts and few-shots."""

    def __init__(self) -> None:
        super().__init__(qiskit_config)

    def self_reflection(
        self, func: str, feedback: str, model: ModelAdapter, temperature: float
    ) -> str:
        return self.generate_self_reflection(
            func=func,
            feedback=feedback,
            model=model,
            temperature=temperature,
        )
