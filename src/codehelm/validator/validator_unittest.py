import json
import os
import sys
import unittest
from pathlib import Path

from codehelm.utils.utils import _build_test_file, extract_code_from_llm_response
from codehelm.validator.validator_base import BaseValidator


class ValidatorUnitTest(BaseValidator):
    def __init__(self):
        self.sample_num = None
        self.test = None
        self.test_format = None
        self.entry_point = None

    def setup(self, *, sample_dir, sample_num, test, test_format, entry_point, prompt=None):
        self._base_sample_dir = sample_dir
        self.sample_num = sample_num
        self.test = test
        self.test_format = test_format
        self.entry_point = entry_point
        self._attempt = 0

        os.makedirs(sample_dir, exist_ok=True)
        Path(os.path.join(sample_dir, "unittest.py")).write_text(test)
        if prompt is not None:
            Path(os.path.join(sample_dir, "prompt.txt")).write_text(prompt)

    def validate(self, raw):
        if not isinstance(raw, str):
            raw = str(raw.last_output())
        unittest_file_pattern = "final_*.py"
        active_dir = self._active_dir()

        Path(os.path.join(active_dir, "raw_output.py")).write_text(raw)
        code = extract_code_from_llm_response(raw)
        Path(os.path.join(active_dir, "code.py")).write_text(code)
        # combined file used by test discovery
        final = _build_test_file(code, self.test, self.test_format, self.entry_point)
        Path(os.path.join(active_dir, f"final_testing_{self.sample_num}.py")).write_text(final)

        result = self._validate(active_dir, unittest_file_pattern)
        self._attempt += 1
        return result

    def _validate(self, active_dir, unittest_file_pattern):
        sample_dir = Path(active_dir)
        result = _run_sample_test(sample_dir, unittest_file_pattern)
        not_run = result.testsRun == 0
        passed = (not not_run) and result.wasSuccessful()
        category = classify_failure(sample_dir, result)

        failure_info = [msg for _, msg in result.failures + result.errors]

        summary = {
            "passed": "not_run" if not_run else passed,
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "failure_category": category,
            "failure_details": [msg for _, msg in result.failures],
            "error_details": [msg for _, msg in result.errors],
        }
        with open(sample_dir / "validator_result.json", "w") as f:
            json.dump(summary, f, indent=4)

        return passed, failure_info
    
    @staticmethod
    def get_summary(results_dir, test_folder_pattern="sample_*") -> dict:
        """Collate already-written ``validator_result.json`` files into a top-level
        ``summary.json``.

        Call this *after* generation is complete.  Each per-sample result must have
        been written by :func:`run_and_save_sample` beforehand.
        """
        results_dir = Path(results_dir)
        if "*" in test_folder_pattern:
            result_folders = sorted(
                (x for x in results_dir.rglob(pattern=test_folder_pattern) if x.is_dir()),
                key=lambda x: int(x.name.split("_")[-1]),
            )
        else:
            result_folders = [Path(os.path.join(results_dir, test_folder_pattern))]

        results = []
        categories = []
        for sample_dir in result_folders:
            result_file = sample_dir / "validator_result.json"
            with open(result_file) as f:
                sample_summary = json.load(f)
            category = sample_summary["failure_category"]
            categories.append(category)
            if category != "not_run":
                results.append(1 if sample_summary["passed"] is True else 0)

        by_category = {}
        for cat in categories:
            by_category[cat] = by_category.get(cat, 0) + 1

        model_summary = {
            "passed": sum(results),
            "total_run": len(results),
            "by_category": by_category,
            "samples": [{"sample": i, "category": cat} for i, cat in enumerate(categories)],
        }
        with open(results_dir / "summary.json", "w") as f:
            json.dump(model_summary, f, indent=4)

        not_run_count = by_category.get("not_run", 0)
        print(f"results: {sum(results)}/{len(results)} passed ({not_run_count} not run)")
        return model_summary

def classify_failure(sample_dir, result):
    from codehelm.utils.patterns import V1_PATTERNS

    if result.testsRun == 0:
        return "not_run"
    if result.wasSuccessful():
        return "passed"

    # static check first catches v1 api use before inspecting tracebacks
    try:
        code = (sample_dir / "code.py").read_text()
        for pattern in V1_PATTERNS:
            if pattern in code:
                return "v1_api"
    except FileNotFoundError:
        pass

    all_messages = " ".join(msg for _, msg in result.errors + result.failures)
    if "ImportError" in all_messages or "ModuleNotFoundError" in all_messages:
        return "import_error"
    if "AttributeError" in all_messages or "TypeError" in all_messages:
        return "api_error"
    if "AssertionError" in all_messages:
        return "logic_error"
    return "unknown_error"


def _run_sample_test(sample_dir, unittest_file_pattern):
    """Run unittest for a single sample and return results."""
    # Clear any cached modules whose file lives anywhere inside the sample
    # directory tree (including parent sample dirs for retry sub-directories).
    # Walking up to the first non-sample ancestor ensures that a retry run
    # does not collide with the cached module from the base pass.
    sample_dir_resolved = sample_dir.resolve()
    # Evict from the nearest "sample_N" ancestor downward so that retry dirs
    # don't inherit stale modules from the base-pass directory.
    evict_root = sample_dir_resolved
    for parent in sample_dir_resolved.parents:
        if parent.name.startswith("sample_"):
            evict_root = parent
            break

    for key, module in list(sys.modules.items()):
        # Evict by name: any module whose top-level name matches the
        # "final_testing_*" pattern must be cleared unconditionally so
        # that a module written by a previous test run (potentially in a
        # different results directory) does not shadow the current one.
        if key.startswith("final_testing_"):
            del sys.modules[key]
            continue
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        try:
            if Path(module_file).resolve().is_relative_to(evict_root):
                del sys.modules[key]
        except (OSError, RuntimeError):
            continue

    with open(sample_dir / "unittest_output.txt", "w") as f:
        loader = unittest.TestLoader()
        suite = loader.discover(
            start_dir=str(sample_dir), pattern=unittest_file_pattern
        )
        runner = unittest.TextTestRunner(stream=f)
        result = runner.run(suite)

    return result

'''
def get_tests(
    results_dir, test_folder_pattern="sample_*", unittest_file_pattern="final_*.py"
):
    """Run and save every sample, then return the collated summary.

    Retained for backwards compatibility.  New code should call
    :func:`run_and_save_sample` per sample and :func:`get_summary` at the end.
    """
    results_dir = Path(results_dir)
    if "*" in test_folder_pattern:
        result_folders = sorted(
            (x for x in results_dir.rglob(pattern=test_folder_pattern) if x.is_dir()),
            key=lambda x: int(x.name.split("_")[-1]),
        )
    else:
        result_folders = [Path(os.path.join(results_dir, test_folder_pattern))]

    for sample_dir in tqdm(result_folders):
        run_and_save_sample(sample_dir, unittest_file_pattern)

    return ValidatorUnitTest.get_summary(results_dir, test_folder_pattern)
'''