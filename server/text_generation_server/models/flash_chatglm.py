from typing import Optional

import torch
import torch.distributed
from opentelemetry import trace
from transformers import AutoTokenizer

from text_generation_server.models import FlashCausalLM
from text_generation_server.models.custom_modeling.flash_chatglm_modeling import (
    ChatGLMConfig,
    FlashChatGLMForConditionalGeneration,
)
from text_generation_server.utils import (
    Weights,
    initialize_torch_distributed,
    weight_files,
)

tracer = trace.get_tracer(__name__)


class FlashChatGLM(FlashCausalLM):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ):
        self.process_group, rank, world_size = initialize_torch_distributed()
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
            dtype = torch.float16 if dtype is None else dtype
        else:
            raise NotImplementedError("FlashChatGLM is only available on GPU")

        # TODO: use `TokenizerFast` instead
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
            encode_special_tokens=True,
        )

        config = ChatGLMConfig.from_pretrained(model_id, revison=revision)
        config.quantize = quantize

        torch.distributed.barrier(group=self.process_group)
        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(filenames, device, dtype, process_group=self.process_group)
        if config.quantize in ["gqtq", "awq"]:
            weights._set_gptq_params(model_id)

        model = FlashChatGLMForConditionalGeneration(config, weights)
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            num_layers=model.num_layers,
            num_kv_heads=model.num_kv_heads,
            head_size=model.head_size,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
        )
