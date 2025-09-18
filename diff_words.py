import re
from collections import  Counter

def remove_common_words(text1, text2):
    word1  = re.findall(r"\w+", text1)
    word2 = re.findall(r"\w+",text2)

    c1= Counter(w.lower() for w in word1)
    c2 = Counter(w.lower() for w in word2)
    common = c1& c2 

    rm_c = dict(common)
    clean1 = []
    for w in word1:
        lw = w.lower()
        if lw in rm_c and rm_c[lw] > 0:
            rm_c[lw] -= 1
        else:
            clean1.append(w)

    rm_c = dict(common)
    clean2 = []
    for w in word2:
        lw = w.lower()
        if lw  in rm_c and rm_c[lw] > 0:
            rm_c[lw] -= 1
        else:
            clean2.append(w)

    return " ".join(clean1), " ".join(clean2)

def queue_common_words(text1:str, text2:str):
    text1 = text1.lower()
    text2 = text2.lower()
    word1  = re.findall(r"\w+", text1)
    word2 = re.findall(r"\w+",text2)
    
        