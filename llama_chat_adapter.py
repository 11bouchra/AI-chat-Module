import os
class LlamaChat:

 def __init__(self, repo_id: str, fileglob:str, filename:str, model_dir:str, n_ctx: int = 0,
             verbose: bool = True) -> None:

    try:
        # This will use the model we've already downloaded and cached locally
        self.model_path = os.path.join(model_dir, filename)
        self.llm = Llama(model_path=self.model_path, 
                         n_ctx=n_ctx,
                         n_gpu_layers=-1,
                         verbose=verbose)
    except:
        try:
            # This will download the model from the repo and cache it locally
            # Handy if we didn't download during install
            self.model_path = os.path.join(model_dir, fileglob)
            self.llm = Llama.from_pretrained(repo_id=repo_id,
                                             filename=fileglob,
                                             n_ctx=n_ctx,
                                             n_gpu_layers=-1,
                                             verbose=verbose,
                                             cache_dir=model_dir,
                                             chat_format="llama-2")
        except:
            self.llm        = None
            self.model_path = None


def do_chat(self, prompt: str, system_prompt: str=None, **kwargs) -> \
        Union[CreateChatCompletionResponse, Iterator[CreateChatCompletionStreamResponse]]:
    """ 
    Generates a response from a chat / conversation prompt
    params:
        prompt:str	                    The prompt to generate text from.
        system_prompt: str=None         The description of the assistant
        max_tokens: int = 128           The maximum number of tokens to generate.
        temperature: float = 0.8        The temperature to use for sampling.
        grammar: Optional[LlamaGrammar] = None
        functions: Optional[List[ChatCompletionFunction]] = None,
        function_call: Optional[Union[str, ChatCompletionFunctionCall]] = None,
        stream: bool = False            Whether to stream the results.
        stop: [Union[str, List[str]]] = [] A list of strings to stop generation when encountered.
    """

    if not system_prompt:
        system_prompt = "You're a helpful assistant who answers questions the user asks of you concisely and accurately."

    completion = self.llm.create_chat_completion(
                    messages=[
                        ChatCompletionRequestSystemMessage(role="system", content=system_prompt),
                        ChatCompletionRequestUserMessage(role="user", content=prompt),
                    ],
                    **kwargs) if self.llm else None

    return completion
