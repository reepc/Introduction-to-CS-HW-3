import os
from typing import Union

import datasets
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase


class CustomDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
    
    def __getitem__(self, index):
        raise NotImplementedError("This method implemente based on your data.")
    
    def __len__(self):
        raise NotImplementedError("This method returns your data's number.")


# For example, this is a summary dataset which it's data is SamSum dataset
class Samsum(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        data_path: Union[str, os.PathLike],
        split: str = "train",
        **kwargs
    ):
        super().__init__()
        # Load data using datasets library (You can load data in other ways)
        data = datasets.load_dataset(data_path, split=split)
        
        # This prompt is the thing you enter when you talk to chatgpt
        self.prompt = (
            f"Summarize this dialog:\n{{dialog}}\n---\nSummary:\n"
        )
        self.tokenizer = tokenizer
        self.data = data.map(self.apply_prompt_template, remove_columns=list(data.features))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        # You need turn text to integer which called tokenize first
        prompt = self.tokenizer.encode(self.tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        summary = self.tokenizer.encode(sample["summary"] +  self.tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt + summary,
            "attention_mask" : [1] * (len(prompt) + len(summary)),
            "labels": [-100] * len(prompt) + summary,
        }

        return sample
    
    def apply_prompt_template(self, sample):
        return {
            "prompt": self.prompt.format(dialog=sample["dialogue"]),
            "summary": sample["summary"],
        }