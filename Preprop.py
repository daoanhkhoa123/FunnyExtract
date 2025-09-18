import torch
from sentence_transformers import SentenceTransformer
from typing import Callable

class Preprop:
    def __init__(self, prepropfn:Callable, aggfn:Callable|None=None, encoder_name="bkai-foundation-models/vietnamese-bi-encoder", device=None) -> None:
        self.prepropfn = prepropfn
        self.aggfn = aggfn if aggfn is not None else lambda x,y : (x,y)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.encoder = SentenceTransformer(encoder_name, device=self.device)

    def __call__(self, text1, text2, batch_size, verbose = True):
        t1, t2 = self.prepropfn(text1,text2)
        v1 =  self.encoder.encode(t1, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=verbose)
        v2 =  self.encoder.encode(t2, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=verbose)
        return self.aggfn(v1,v2)
        
    