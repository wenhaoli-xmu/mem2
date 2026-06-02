import json
from torch.utils.data import Dataset
from datasets import load_dataset

import torch



from typing import Union


def pack_lm(text: Union[str, torch.Tensor]):
    return {
        "text": text,
        "task_type": "language modeling"
    }


class Github(Dataset):
    """
    github.256k   # for qwen3 only
    github.40k    # for qwen3 only
    github.128k   # for llama3 only
    """

    def __init__(self, split, max_length=None):
        with open(f'data/github-{split}-00000.json', 'r') as f:
            self.data = [json.loads(line) for line in f.readlines()]

    def __getitem__(self, index):
        return pack_lm(torch.tensor(self.data[index]['input_ids'], dtype=torch.int64))
    
    def __len__(self):
        return len(self.data)
    

class PG19(Dataset):
    """
    pg19.1m
    pg19.256k
    pg19.128k
    pg19.prolong
    """

    def __init__(self, max_length=None):
        import os
        assert os.path.exists(os.environ['SPOTLIGHT_PG19_PATH']), f"Please download pg19.json first."

        if max_length == 'prolong':
            os.environ['SPOTLIGHT_PG19_PATH'] = os.environ['SPOTLIGHT_PG19_PATH'].replace("pg19.json", "pg19-prolong.json")

        with open(os.environ['SPOTLIGHT_PG19_PATH'], "r") as f:
            self.data = [json.loads(line) for line in f.readlines()]
        
        if max_length == '1m':
            self.maximum = 1024 * 1024
        elif max_length == '256k':
            self.maximum = 256 * 1024
        elif max_length == '128k':
            self.maximum = 128 * 1024
        else:
            self.maximum = None

    def __getitem__(self, index):
        if self.maximum is not None:
            text = self.data[index]['text'][:self.maximum]
        else:
            text = self.data[index]['text']

        return pack_lm(text)
    
    def __len__(self):
        return len(self.data)
    

class ProofPile(Dataset):
    """
    proof-pile
    proof-pile.1m
    proof-pile.256k
    """

    def __init__(self, max_length=None):
        import os
        assert os.path.exists(os.environ['SPOTLIGHT_PROOFPILE_PATH']), f"Please download proof-pile.json first."
        with open(os.environ['SPOTLIGHT_PROOFPILE_PATH'], "r") as f:
            self.data = [json.loads(line) for line in f.readlines()]

        if max_length == '1m':
            self.maximum = 1024 * 1024
        elif max_length == '256k':
            self.maximum = 256 * 1024
        else:
            self.maximum = None

    def __getitem__(self, index):
        if self.maximum is not None:
            text = self.data[index]['text'][:self.maximum]
        else:
            text = self.data[index]['text']
        return pack_lm(text)
    
    def __len__(self):
        return len(self.data)


class CodeParrot(Dataset):
    """
    code-parrot
    code-parrot.1m
    code-parrot.256k
    code-parrot.128k
    """

    def __init__(self, max_length=None):
        import os
        assert os.path.exists(os.environ['SPOTLIGHT_CODEPARROT_PATH']), f"Please download codeparrot.json first."
        with open(os.environ['SPOTLIGHT_CODEPARROT_PATH'], "r") as f:
            self.data = [json.loads(line) for line in f.readlines()]

        if max_length == '1m':
            self.maximum = 1024 * 1024
        elif max_length == '256k':
            self.maximum = 256 * 1024
        elif max_length == '128k':
            self.maximum = 128 * 1024
        elif max_length is None:
            pass
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        if self.maximum is not None:
            text = self.data[index]['text'][:self.maximum]
        else:
            text = self.data[index]['text']
        return pack_lm(text)
    
    def __len__(self):
        return len(self.data)


CORPUS_MAPPING = {
    # language modeling
    "pg19": PG19,
    "proof-pile": ProofPile,
    "code-parrot": CodeParrot,
    "github": Github
}


def get_corpus(ds):
    ds = ds.lower().replace(" ", "")
    dataset_name, *args = ds.split(".")

    for name, data_class in CORPUS_MAPPING.items():
        if name == dataset_name:
            return data_class(*args)
    
    raise NotImplementedError(dataset_name)
