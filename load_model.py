import os
from typing import Union

from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_cpp import Llama


def load_taide():
    """
    Request access first here: https://huggingface.co/taide/TAIDE-LX-7B-Chat
    """
    model = AutoModelForCausalLM.from_pretrained("taide/TAIDE-LX-7B-Chat", device_map="auto") # You can change device_map, it's mean which device you are going to put your model
    tokenizer = AutoTokenizer.from_pretrained("taide/TAIDE-LX-7B-Chat")
    return model, tokenizer

def load_llama(gguf_file_path: Union[os.PathLike, str], using_llama_cpp: bool = True):
    """
    You need to convert original llama model to .gguf format using this function.
    You also need to access the Llama models' access first.
    """
    if using_llama_cpp:
        model = Llama(
            model_path=gguf_file_path,
            n_gpu_layers=-1, # Need you to change by yourself, if -1, all the layers will be offloaded to GPU
            n_ctx=0,
            chat_format="llama-3",
            verbose=False,
        )
    
        return model
    
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    
    return model, tokenizer