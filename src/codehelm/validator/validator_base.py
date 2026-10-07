import os
from abc import ABC, abstractmethod


class BaseValidator(ABC):
    def __init__(self):
        self._attempt = 0
        self._base_sample_dir = None

    @abstractmethod
    def setup(self, **kwargs):
        """
        Setup routine if a validator needs per-iteration updates
        """
    
    @abstractmethod
    def validate(self, raw):
        """
        Validation logic

        raw - either a string or mellea context
        """
    
    def _active_dir(self):
        """Return the directory for the current attempt, creating it if needed."""
        if self._base_sample_dir:
            if self._attempt == 0:
                return self._base_sample_dir
            retry_dir = os.path.join(self._base_sample_dir, f"retry_{self._attempt - 1}")
            os.makedirs(retry_dir, exist_ok=True)
            return retry_dir
        else:
            raise ValueError("_base_sample_dir not set")

