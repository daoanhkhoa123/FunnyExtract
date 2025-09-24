from typing import List, Literal, Tuple
from dataclasses import dataclass

from transformers import AutoTokenizer, AutoModel
import torch
from torch import Tensor, nn

DEFAULT_LLM=  "bkai-foundation-models/vietnamese-bi-encoder"

@dataclass
class DecodeStyleParams:
    llm_name:str
    num_heads:int
    num_atnns:int
    dropout:float
    huggingface_device:str
    

class HuggingFaceEncoder_notrain:
    POOLING_TYPE = Literal["token", "mean","sentence"]
    def __init__(self, llm_name = DEFAULT_LLM, device = "cpu") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.model = AutoModel.from_pretrained(llm_name).to(device)
        self.device = device

    @property
    def hidden_dim(self):
        return self.model.config.hidden_size

    @staticmethod
    def mean_pooling(input:Tensor, attn_mask:Tensor) -> Tensor:
        attn_mask = attn_mask.unsqueeze(-1).expand(input.size()).float()
        return torch.sum(input * attn_mask, 1)/torch.clamp(torch.sum(attn_mask,1), min=1e-9)

    def __call__(self, batch_str:List[str], pooling:POOLING_TYPE="token") -> Tensor:
        tokens =self.tokenizer(batch_str, padding=True, truncation=True, return_tensors='pt')
        tokens = {k:v.to(self.device) for k, v in tokens.items()}
        with torch.no_grad():
            embeddings = self.model(**tokens)

        if pooling == "token":
            return embeddings[0]
        if pooling == "mean":
            return self.mean_pooling(embeddings[0], tokens["attention_mask"])
        if pooling == "sentence":
            return embeddings[1]
        raise ValueError()

class BiMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln2l = nn.LayerNorm(embed_dim)
        self.mlpl = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.drop2l = nn.Dropout(dropout)
        self.ln1l = nn.LayerNorm(embed_dim)
        self.attnl = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.drop1l = nn.Dropout(dropout)

        self.ln2r = nn.LayerNorm(embed_dim)
        self.mlpr = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.drop2r = nn.Dropout(dropout)
        self.ln1r = nn.LayerNorm(embed_dim)
        self.attnr = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.drop1r = nn.Dropout(dropout)

    def forward(self, context: Tensor, response: Tensor) -> Tuple[Tensor, Tensor]:
        cntx = self.attnl(self.ln1l(context), response, response)[0]
        context = context + self.drop1l(cntx)
        context = context + self.drop2l(self.mlpl(self.ln2l(context)))

        resp = self.attnr(self.ln1r(response), context, context)[0]
        response = response + self.drop1r(resp)
        response = response + self.drop2r(self.mlpr(self.ln2r(response)))

        return context, response


class DecoderStyle(nn.Module):
    def __init__(self, arg:DecodeStyleParams) -> None:
        super().__init__()
        self.encoder = HuggingFaceEncoder_notrain(arg.llm_name, arg.huggingface_device)
        
        embed_dim = self.encoder.hidden_dim
        self.attns = nn.ModuleList(
            [BiMultiHeadAttention(embed_dim, arg.num_heads, arg.dropout)
             for _ in range (arg.num_atnns)]
        )

        self.classifer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3)
        )


    def forward(self, batch_texts:List[str]) -> Tensor:
        """ Batch is a list of str, first half are all contexts, second are all respones.
        This is for batch supporting style of hugging face model 
        
        self.encoder is not trainable
        """
        half = len(batch_texts)//2
        batch_emb = self.encoder(batch_texts)
        contexts = batch_emb[:half]
        responses = batch_emb[half:]

        for layer in self.attns:
            contexts, responses = layer(contexts, responses)

        pooled = responses.mean(dim=1)
        return self.classifer(pooled)