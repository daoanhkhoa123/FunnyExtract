import torch
from torch.utils.data import Dataset
import pandas as pd 

from typing import Callable, Tuple

class VillahuDecoderStyle(Dataset):
    def __init__(self, df_path:str, preprop:Callable[[str,str], Tuple[str,str]]) -> None:
        super().__init__()
        self.df = pd.read_csv(df_path)
        self.preprop = preprop

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index) -> Tuple[Tuple[str, str], int]:
        context = str(self.df.at[index, "context"])
        response = str(self.df.at[index, "response"])
        return self.preprop(context, response), int(self.df.iloc[index]["label"])
    


def collate_fn(batch):
    pair, label = zip(*batch)
    contexts, responses = zip(*pair)

    return list(contexts) + list(responses), label


