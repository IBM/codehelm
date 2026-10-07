# Cross-checked against:
# https://docs.quantum.ibm.com/migration-guides/qiskit-1.0
# https://docs.quantum.ibm.com/migration-guides/qiskit-2.0
V1_PATTERNS = [
    # Removed top-level execute() function (v1)
    "from qiskit import execute",
    "execute(",
    # Removed global Aer/BasicAer instances (v1)
    "from qiskit import Aer",
    "from qiskit import BasicAer",
    "Aer.get_backend(",
    "BasicAer",
    "qiskit.providers.basicaer",
    # qiskit.tools submodules removed (v1)
    "from qiskit.tools",
    "qiskit.tools",
    # BackendV1 and provider model classes removed (v2)
    "from qiskit.providers import BackendV1",
    "BackendV1",
    "from qiskit.providers.models import",
    # qiskit.pulse module removed entirely (v2)
    "import qiskit.pulse",
    "from qiskit.pulse",
    # qiskit.scheduler module removed (v2)
    "import qiskit.scheduler",
    "from qiskit.scheduler",
    # qiskit.circuit.classicalfunction removed (v2)
    "from qiskit.circuit.classicalfunction",
    # QuantumCircuit methods removed (v2)
    ".c_if(",
    ".add_calibration(",
    # Compiler functions sequence/schedule removed (v2)
    "from qiskit.compiler import sequence",
    "from qiskit.compiler import schedule",
]
