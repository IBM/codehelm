import os


def resolve_env_vars(config: dict) -> dict:
    """Recursively replace ``"env:VAR_NAME"`` string values with the
    corresponding environment variable.

    Raises ``KeyError`` if a referenced variable is not set, so misconfigured
    runs fail early with a clear message rather than passing ``None`` to an API.
    """
    for key, value in config.items():
        if isinstance(value, dict):
            resolve_env_vars(value)
        elif isinstance(value, str) and value.startswith("env:"):
            var_name = value[4:]
            env_value = os.environ.get(var_name)
            if env_value is None:
                raise KeyError(
                    f"Config references environment variable '{var_name}' "
                    f"(via \"{value}\") but it is not set."
                )
            config[key] = env_value
    return config


def resolve_folder_savedir(default_results_prefix="results"):
    """
    Resolve a unique directory name by checking if the default exists.
    If it exists, append _v2, _v3, etc. until a non-existing directory is found.

    Args:
        default_results_prefix: The base directory name (default: "results")

    Returns:
        str: A unique directory path that doesn't exist yet
    """
    savedir = default_results_prefix

    # Check if the default directory exists
    if os.path.exists(savedir):
        version = 2
        # Keep incrementing version until we find a non-existing directory
        while os.path.exists(f"{default_results_prefix}_v{version}"):
            version += 1
        savedir = f"{default_results_prefix}_v{version}"

    return savedir


def pull_python_snippet(text, start_char="```python\n", end_char="```\n\n"):
    """
    Function for extracting Python code from markdown code blocks.
    Kept for backward compatibility.

    Args:
        text: The text containing the code block
        start_char: The starting delimiter
        end_char: The ending delimiter

    Returns:
        str: Extracted code
    """
    start_idx = text.find(start_char) + len(start_char)
    end_idx = text.rfind(end_char)  # rfind needed for models that repeat closing fence
    return text[start_idx:end_idx]


def extract_code_from_llm_response(text):
    """
    Extract code from LLM response that may be wrapped in markdown code blocks.
    Handles various formats:
    - ```python ... ```
    - ```py ... ```
    - ``` ... ```
    - Plain code without wrapping

    Args:
        text: The LLM response text

    Returns:
        str: Extracted code, or original text if no code block found
    """
    import re

    # Try to find code blocks with various language identifiers
    patterns = [
        r"```python\n(.*?)```",
        r"```py\n(.*?)```",
        r"```\n(.*?)```",
        r"```(.*?)```",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            # Remove any trailing newlines or extra backticks
            code = code.rstrip("`\n ")
            return code

    # If no code block found, check if the text itself looks like code
    # (contains common Python keywords)
    python_keywords = ["def ", "class ", "import ", "from ", "if ", "for ", "while "]
    if any(keyword in text for keyword in python_keywords):
        return text.strip()

    # Return original text as fallback
    return text.strip()


def _build_test_file(code, test, test_format, entry_point=None):
    """Combine generated code and test into a file discoverable by unittest.

    Args:
        code: The extracted model-generated code.
        test: The raw test string from the dataset.
        test_format: ``"unittest"`` — test is already a ``TestCase`` class;
                     ``"human_eval"`` — test is a ``check(candidate)`` function
                     that must be wrapped in a ``TestCase``.
        entry_point: Function name to pass to ``check()`` (required for
                     ``human_eval`` format).
    """
    if test_format == "human_eval":
        if not entry_point:
            raise ValueError("entry_point is required for human_eval test_format")
        wrapper = (
            "\nimport unittest\n"
            "class TestSolution(unittest.TestCase):\n"
            "    def test_solution(self):\n"
            f"        check({entry_point})\n"
            "\nif __name__ == '__main__':\n"
            "    unittest.main()\n"
        )
        return code + "\n" + test + wrapper
    else:
        # "unittest" format: test is already a TestCase, just concatenate
        return code + "\n" + test


