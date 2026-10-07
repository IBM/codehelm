from typing import Any, Literal

from tqdm import tqdm

from codehelm.models.models import CoderModel, ModelConfig


def _require_huggingface():
    """Raise a helpful error if the huggingface extra is not installed."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "The 'huggingface' extra is required to use HuggingFaceModel. "
            "Install it with: pip install 'codehelm[huggingface]'"
        ) from e

class HuggingFaceModelConfig(ModelConfig):

    type: Literal["codehelm.models.huggingface.HuggingFaceModel"]

class HuggingFaceModel(CoderModel):
    config: HuggingFaceModelConfig

    # loads model config and initialises the model
    def __init__(self, config):
        _require_huggingface()
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Mistral3ForConditionalGeneration,
        )
        super().__init__(config=config)
        # mistral3 not supported by AutoModelForCausalLM

        model_cls = (
            Mistral3ForConditionalGeneration
            if "mistral"
            in self.config.pretrained_model_name_or_path.lower()
            else AutoModelForCausalLM
        )
        self.model = model_cls.from_pretrained(
            self.config.pretrained_model_name_or_path,
            trust_remote_code=True,
            device_map=self.config.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.pretrained_model_name_or_path
        )

    @classmethod
    def from_model_and_tokenizer(cls, model, tokenizer):
        """Wrap already-instantiated model and tokenizer without reloading weights."""
        instance = cls.__new__(cls)
        instance.model = model
        instance.tokenizer = tokenizer
        return instance

    @classmethod
    def config_model(cls) -> type[HuggingFaceModelConfig]:
        return HuggingFaceModelConfig

    # sends a prompt to the model and returns the raw response
    def generate(self, prompt, verbose=False, apply_chat_template=True):
        if apply_chat_template:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )

        model_inputs: Any = self.tokenizer(prompt, return_tensors="pt").to(
            self.model.device
        )

        generated_ids = self.model.generate(
            **model_inputs, max_new_tokens=512, do_sample=False
        )
        # strip input tokens from output
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]
        if verbose:
            print(response)
        return response

    def logits_generate(
        self, prompt, top_k=1, apply_chat_template=True, max_tokens=512
    ):
        # Mistral3ForConditionalGeneration doesn't support returning the logits.
        # implement a manual forward pass
        eos_token_id = self.tokenizer.eos_token_id
        if apply_chat_template:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = self.tokenizer(text, return_tensors="pt").to(
                self.model.device
            )
        else:
            model_inputs = self.tokenizer(prompt, return_tensors="pt").to(
                self.model.device
            )

        input_ids = model_inputs["input_ids"]
        past_key_values = None
        scores = []
        top_k_info = []  # Store top k tokens and their logits for each position
        import torch
        with torch.no_grad():
            for _ in tqdm(range(max_tokens)):
                out = self.model(
                    input_ids=input_ids[:, -1:]
                    if past_key_values is not None
                    else input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                next_token_logits = out.logits[:, -1, :]
                past_key_values = out.past_key_values

                # Get top k tokens and their logits
                topk_logits, topk_indices = torch.topk(
                    next_token_logits, k=min(top_k, next_token_logits.shape[-1]), dim=-1
                )

                # Store top k information for this position (convert to JSON-serializable types)
                top_k_info.append(
                    {
                        "logits": topk_logits.to(torch.float32)
                        .detach()
                        .cpu()
                        .numpy()
                        .tolist(),
                        "token_ids": topk_indices.detach().cpu().numpy().tolist(),
                        "tokens": [
                            self.tokenizer.decode([token_id])
                            for token_id in topk_indices[0].tolist()
                        ],
                    }
                )

                # Use the top 1 token for generation (argmax)
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=-1)

                scores.append(
                    next_token_logits.to(torch.float32).detach().cpu().numpy()
                )

                if eos_token_id is not None and torch.all(next_token == eos_token_id):
                    break
        tokens_to_decode = input_ids[:, model_inputs["input_ids"].shape[-1] :]
        response = self.tokenizer.batch_decode(
            tokens_to_decode, skip_special_tokens=True
        )[0]
        tokens_to_decode = tokens_to_decode.reshape(-1, 1)
        split_tokens = self.tokenizer.batch_decode(
            tokens_to_decode, skip_special_tokens=True
        )
        return response, scores, split_tokens, top_k_info
