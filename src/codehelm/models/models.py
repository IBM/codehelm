from abc import ABC, abstractmethod
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ModelConfig(BaseModel):
    """Base configuration for Model"""

    type: Annotated[str, Field(description="Full class name")]
    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        use_enum_values=True,
        arbitrary_types_allowed=True,
        validate_default=True,
    )


class CoderModel(ABC):
    # loads model config and initialises the model (rits or huggingface)
    def __init__(self, config):
        if isinstance(config, dict):
            try:
                validated_config = self.config_model().model_validate(config)
                self.config = validated_config
            except ValidationError as e:
                error_message = f"Config validation failed: {e}"
                raise ValueError(error_message) from e
        else:
            self.config = config

    @classmethod
    def config_model(cls) -> type[ModelConfig]:
        """
        Return the Pydantic model class for this model class

        :return: Pydantic model class
        """
        return ModelConfig

    @abstractmethod
    def generate(self, prompt):
        """
        Single generation
        """
        raise NotImplementedError
