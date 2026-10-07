import json
import logging
import os
import signal
import threading
import traceback

logger = logging.getLogger(__name__)

from contextlib import contextmanager
from pathlib import Path

from codehelm.utils.utils import extract_code_from_llm_response
from codehelm.validator.validator_base import BaseValidator

EXECUTION_TIMEOUT_SECONDS = 60

@contextmanager
def timeout(seconds: int):
    """
    Context manager to enforce a timeout on code execution.

    Uses SIGALRM signal to interrupt execution after the specified time limit.
    This prevents infinite loops and resource exhaustion attacks.

    Args:
        seconds (int): Maximum number of seconds to allow execution

    Raises:
        TimeoutError: If execution exceeds the specified timeout

    Example:
        Use as a context manager to limit execution time:
        with timeout(5):
            result = expensive_computation()

    Note:
        Only works on Unix-like systems (uses signal.SIGALRM).
        Not available on Windows.
    """

    is_main_thread = threading.current_thread() is threading.main_thread()
    if is_main_thread and hasattr(signal, "SIGALRM"):
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Execution exceeded {seconds} seconds")

        # Set the signal handler and alarm
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            # Restore the old handler and cancel the alarm
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        yield

class ValidatorExec(BaseValidator):
    def __init__(self):
        self.sample_num = None
        self.test = None
        self.entry_point = None

    def setup(self, *, sample_dir, sample_num, test, entry_point, prompt=None, **kwargs):
        self._base_sample_dir = Path(sample_dir)
        self.sample_num = sample_num
        self.test = test
        self.entry_point = entry_point
        self._attempt = 0
        os.makedirs(sample_dir, exist_ok=True)

    def validate(self, raw):
        """
        Code based on qiskit-human-eval's test function:
        https://github.com/qiskit-community/qiskit-human-eval/blob/main/scripts/test_solutions.py

        Test a single problem's canonical solution against its test case.

        Executes the problem's canonical solution in a restricted, isolated namespace
        with timeout protection. Combines the prompt (optional), solution, and test code,
        then runs the test's check function against the implemented entry point.

        Args:
            problem (dict[str, Any]): Problem dictionary containing:
                - task_id (str): Unique identifier for the problem
                - prompt (str): Problem description and function signature
                - canonical_solution (str): Reference implementation
                - test (str): Test code with check function
                - entry_point (str): Name of the function to test
            exclude_prompt (bool, optional): If True, don't prepend prompt to execution.
                                            Defaults to False.

        Returns:
            tuple[bool, Optional[str]]: A tuple containing:
                - success (bool): True if test passed, False otherwise
                - error_message (str | None): Error description if failed, None if passed
        """
        if not isinstance(raw, str):
            raw = str(raw.last_output())
        active_dir = self._active_dir()
        self._attempt += 1
        
        code = extract_code_from_llm_response(raw)

        try:
            # Create executable code by combining prompt, solution, and test
            code = code + "\n" + self.test
            Path(os.path.join(active_dir, "code.py")).write_text(code)

            # ⚠️  SECURITY WARNING: exec() Execution
            # This script uses exec() to execute code from the dataset.
            # ONLY run this script on code you trust from version-controlled sources.
            # exec() can execute arbitrary code and is dangerous if used with untrusted input.
            #
            # For this project, this is safe because:
            # - Dataset is version-controlled in Git
            # - All changes require code review
            # - Only runs on trusted developer/CI systems
            #
            # DO NOT use this script with:
            # - User-submitted code
            # - Untrusted data sources
            # - Network-provided solutions
            #
            # See: security-analysis-exec.md for detailed security analysis

            # Build a restricted builtins dict by copying the real builtins and
            # removing functions that could be used to escape the sandbox, access
            # the filesystem, spawn processes, or execute further arbitrary code.
            # Note: this is a defence-in-depth measure — the primary trust boundary
            # is the version-controlled, code-reviewed dataset.
            #
            # Note: __import__ must remain available — Python's 'import' statement
            # calls it internally. Blocking it would break all imports inside exec().
            import builtins as _builtins

            _BLOCKED_BUILTINS = {
                # "open",         # filesystem read/write // needed by qiskitHumanEval/82
                "exec",         # nested arbitrary code execution
                "eval",         # expression-level arbitrary execution
                "compile",      # bytecode compilation (precursor to exec/eval)
                "breakpoint",   # drops into debugger / pdb shell
                "input",        # blocks on stdin in CI; not needed by solutions
            }
            restricted_builtins = {
                k: v
                for k, v in vars(_builtins).items()
                if k not in _BLOCKED_BUILTINS
            }

            # Execute the code in a restricted namespace with timeout protection.
            # Passing __builtins__ explicitly prevents exec() from injecting the
            # full built-in scope automatically.
            namespace: dict[str, Any] = {"__builtins__": restricted_builtins}

            logger.debug(f"Executing code with {EXECUTION_TIMEOUT_SECONDS}s timeout")
            with timeout(EXECUTION_TIMEOUT_SECONDS):
                exec(code, namespace)

            # Run the check function against the implemented function
            entry_point = self.entry_point
            logger.debug(f"Checking entry point: {entry_point}")

            if entry_point not in namespace:
                logger.warning(f'Entry point "{entry_point}" not found in namespace')
                return False, f'Entry point "{entry_point}" not defined'

            namespace["check"](namespace[entry_point])
            logger.info("✓ passed")

            summary = {
                "passed": True,
            }
            with open(self._base_sample_dir / "validator_result.json", "w") as f:
                json.dump(summary, f, indent=4)

            return True, None

        except TimeoutError as e:
            logger.error(f"✗ timed out: {e}")
            return False, f"Execution timeout: {e}"
        except AssertionError as e:

            summary = {
                "passed": False,
            }
            with open(self._base_sample_dir / "validator_result.json", "w") as f:
                json.dump(summary, f, indent=4)

            logger.error(f"✗ assertion failed: {e}")
            return False, f"Test assertion failed: {e}"
        except Exception as e:

            summary = {
                "passed": False,
            }
            with open(self._base_sample_dir / "validator_result.json", "w") as f:
                json.dump(summary, f, indent=4)
            
            # Include full traceback for debugging
            error_detail = traceback.format_exc()
            Path(os.path.join(active_dir, "error_detail.txt")).write_text(error_detail)
            logger.error(f"✗ failed with {type(e).__name__}: {e}")
            logger.debug(f"Full traceback for :\n{error_detail}")
            return False, f"{type(e).__name__}: {e}\n{error_detail}"
    
    @staticmethod
    def get_summary(results_dir, test_folder_pattern="sample_*") -> dict:
        results_dir = Path(results_dir)
        if "*" in test_folder_pattern:
            result_folders = sorted(
                (x for x in results_dir.rglob(pattern=test_folder_pattern) if x.is_dir()),
                key=lambda x: int(x.name.split("_")[-1]),
            )
        else:
            result_folders = [Path(os.path.join(results_dir, test_folder_pattern))]
        
        results = []
        for sample_dir in result_folders:
            result_file = sample_dir / "validator_result.json"
            with open(result_file) as f:
                sample_summary = json.load(f)
            results.append(1 if sample_summary["passed"] is True else 0)
        
        model_summary = {"passed": sum(results)}
        with open(results_dir / "summary.json", "w") as f:
            json.dump(model_summary, f, indent=4)

        print(f"results: {sum(results)}/{len(results)} passed.")
        return model_summary
