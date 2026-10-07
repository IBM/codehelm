
class Feedback:
    """Build a retry prompt by combining the original prompt with unittest failure info."""

    RETRY_TEMPLATE = (
        "Your previous attempt failed the following unittest:\n\n"
        "{failure_info}\n\n"
        "Please fix your code and try again.\n\n"
        "Original prompt:\n{prompt}"
    )
    def __init__(self):
        self.requires_model = False

    def setup(self, sample_dir):
        self._attempt = 0
    
    def repair(self, prompt: str, failure_info: str) -> str:
        """Construct a new prompt that includes the failure feedback.

        Args:
            prompt:       The original generation prompt.
            failure_info: Unittest failure/error output to feed back to the model.

        Returns:
            A combined prompt string ready to pass back to the model.
        """
        return self.RETRY_TEMPLATE.format(prompt=prompt, failure_info=failure_info)
