from tokenizers import Tokenizer
import numpy as np

# Load tokenizer from tokenizer.json (MiniLM-compatible)
tokenizer = Tokenizer.from_file("model/tokenizer.json")

def tokenize(texts):
    if isinstance(texts, str):
        texts = [texts]

    encodings = [tokenizer.encode(text) for text in texts]

    max_len = max(len(enc.ids) for enc in encodings)
    input_ids = []
    attention_mask = []
    token_type_ids = []

    for enc in encodings:
        ids = enc.ids
        mask = [1] * len(ids)
        types = [0] * len(ids)  # single sentence = all 0s

        # Pad to max_len
        padding = [0] * (max_len - len(ids))
        input_ids.append(ids + padding)
        attention_mask.append(mask + padding)
        token_type_ids.append(types + padding)

    return {
        "input_ids": np.array(input_ids, dtype=np.int64),
        "attention_mask": np.array(attention_mask, dtype=np.int64),
        "token_type_ids": np.array(token_type_ids, dtype=np.int64)
    }
