import os
from typing import Iterator, Union
from llama_cpp import (
    ChatCompletionRequestSystemMessage,
    ChatCompletionRequestUserMessage,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
    Llama
)

class LlamaChat:
    def __init__(self, repo_id: str, fileglob:str, filename:str, model_dir:str, n_ctx: int = 0, verbose: bool = True):
        """
        Initialize the Llama model, either from a cached local model or download from repo.
        """
        try:
            self.model_path = os.path.join(model_dir, filename)
            self.llm = Llama(model_path=self.model_path, n_ctx=n_ctx, n_gpu_layers=-1, verbose=verbose)
        except:
            try:
                self.model_path = os.path.join(model_dir, fileglob)
                self.llm = Llama.from_pretrained(repo_id=repo_id,
                                                 filename=fileglob,
                                                 n_ctx=n_ctx,
                                                 n_gpu_layers=-1,
                                                 verbose=verbose,
                                                 cache_dir=model_dir,
                                                 chat_format="llama-2")
            except:
                self.llm = None
                self.model_path = None

    def do_chat(self, prompt: str, system_prompt: str=None, **kwargs) -> Union[CreateChatCompletionResponse, Iterator[CreateChatCompletionStreamResponse]]:
        """Generate response from a chat prompt."""
        if not system_prompt:
            system_prompt = "You're a helpful assistant who answers questions concisely and accurately."
        completion = self.llm.create_chat_completion(
            messages=[
                ChatCompletionRequestSystemMessage(role="system", content=system_prompt),
                ChatCompletionRequestUserMessage(role="user", content=prompt),
            ],
            **kwargs
        ) if self.llm else None
        return completion