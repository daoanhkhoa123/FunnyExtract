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

    def __getitem__(self, index) -> Tuple[Tuple[str, str], torch.Tensor]:
        context = str(self.df.at[index, "context"])
        response = str(self.df.at[index, "response"])
        label = torch.tensor(self.df.iloc[index]["label"], dtype= torch.long)
        return self.preprop(context, response), label
    


def collate_fn(batch):
    pair, labels = zip(*batch)
    contexts, responses = zip(*pair)
    labels = torch.stack(labels)
    
    return list(contexts) + list(responses), labels


