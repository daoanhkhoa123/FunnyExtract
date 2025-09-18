import pandas as pd
import numpy as np
from diff_words import remove_common_words

def preppropfn(df):
  df[["context_clean", "response_clean"]] = df.apply(
    lambda row: pd.Series(remove_common_words(row["context"], row["response"])),
    axis=1
    )
  
  t1 = df["context_clean"].tolist()
  t2 = df["response_clean"].tolist()
  return t1, t2


def axis_cosines(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return np.full_like(v, np.nan, dtype=float)  # undefined for zero vector
    return v / norm 

def aggfn(v1,v2):
  v = v1- v2
  return axis_cosines(v)
