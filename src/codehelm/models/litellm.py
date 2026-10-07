import dataclasses
from typing import Annotated, Literal

import litellm
from pydantic import Field

from codehelm.models.models import CoderModel, ModelConfig


class LiteLLMConfig(ModelConfig):
    type: Literal["codehelm.models.litellm.LiteLLMModel"]
    model_name:   Annotated[str,           Field(description="LiteLLM model string")]
    api_base:     Annotated[str | None,    Field(description="Custom API base URL")] = None
    api_key:      Annotated[str | None,    Field(description="API key override")]    = None
    max_tokens:   Annotated[int,           Field(description="Max tokens")]          = 1024
    temperature:  Annotated[float,         Field(description="Sampling temperature")] = 1.0
    num_comps:    Annotated[int,           Field(description="Number of completions")] = 1
    warn_on_length: Annotated[bool,        Field(description="Print a warning when finish_reason is 'length' (max tokens reached)")] = True
    system_prompt:  Annotated[str | None,  Field(description="System prompt prepended to every request")] = None

class LiteLLMModel(CoderModel):
    config: LiteLLMConfig

    @classmethod
    def config_model(cls):
        return LiteLLMConfig

    def __init__(self, config):
        super().__init__(config)

    @staticmethod
    def _resolve_model(model: str, api_base: str | None) -> str:
        """Prepend 'openai/' when a custom api_base is given.
        This tells LiteLLM to treat the endpoint as an OpenAI-compatible proxy
        and forward the model name as-is to the remote server."""
        if not api_base:
            return model
        if model.startswith("openai/"):
            return model
        return f"openai/{model}"

    def generate(
        self,
        prompt: str | list,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif not isinstance(prompt[0], dict):
            messages = [dataclasses.asdict(m) for m in prompt]
        else:
            messages = prompt
        resolved_system_prompt = system_prompt if system_prompt is not None else self.config.system_prompt
        if resolved_system_prompt is not None:
            if messages and messages[0]["role"] == "system":
                messages = [{"role": "system", "content": resolved_system_prompt + "\n" + messages[0]["content"]}] + messages[1:]
            else:
                messages = [{"role": "system", "content": resolved_system_prompt}] + messages

        resolved_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        resolved_temperature = temperature if temperature is not None else self.config.temperature
        response = litellm.completion(
            model=self._resolve_model(self.config.model_name, self.config.api_base),
            messages=messages,
            max_tokens=resolved_max_tokens,
            temperature=resolved_temperature,
            timeout=120,
            api_base=self.config.api_base,
            api_key=self.config.api_key,
        )
        choice = response.choices[0]
        if self.config.warn_on_length and choice.finish_reason == "length":
            import warnings
            warnings.warn(
                f"Response for model '{self.config.model_name}' hit the max_tokens limit "
                f"({resolved_max_tokens}). The output may be truncated or empty. "
                "Consider increasing max_tokens in your config.",
                RuntimeWarning,
                stacklevel=2,
            )
        message = choice.message
        # Reasoning models (e.g. o4-mini) may return None for `content` and put
        # the actual answer in `reasoning_content` instead.
        text = message.content or message.get("reasoning_content") or ""
        if verbose:
            print(text)
        return text

