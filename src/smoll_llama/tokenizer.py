import mlx.core as mx
from typing import Optional

class Tokenizer(object):
    def __init__(self):
        self.tok_to_ids = None
        self.ids_to_tok = None
        self.special_tokens = dict()
        self.init_tokenizer()

    def init_tokenizer(self):
        with open("data/train_texts.txt", "r") as f:
            input_text = f.readlines()
        text = [text for text in input_text if text != "\n"]
            
        self.tok_to_ids = dict()
        self.ids_to_tok = dict()
        tokens = sorted(set(" ".join(text).lower()))
        for idx, key in enumerate(tokens):
            self.tok_to_ids[key] = idx
            self.ids_to_tok[idx] = key

        self.add_token("<start>", True, "start")
        self.add_token("<stop>", True, "stop")
        self.add_token("<pad>", True, "pad")
        self.add_token("<unk>", True, "unk")

    def add_token(
        self, 
        tok, 
        is_special: bool = False, 
        special_token_type: Optional[str] = None
    ) -> None:
        if tok in self.tok_to_ids:
            raise ValueError(f"token exists, id: {self.tok_to_ids[tok]}")
        
        max_id = max(self.tok_to_ids.values()) + 1
        self.tok_to_ids[tok] = max_id
        self.ids_to_tok[max_id] = tok
        
        if is_special: 
            self.special_tokens[special_token_type] = max_id

    def preprocess_string(
        self, 
        string: str, 
        add_special_tokens: bool=True,
    ) -> mx.array:
        unk = self.special_tokens["unk"]
        out = [self.tok_to_ids.get(s, unk) for s in string.lower()]
        if add_special_tokens:
            assert len(self.special_tokens) > 0
            out = [self.special_tokens["start"]] + out + [self.special_tokens["stop"]]
        return mx.array(out)

    def pad_tokens(self, toks: mx.array, n: int) -> mx.array:
        to_add = n - toks.shape[0]
        if to_add == 0:
            return toks
        if to_add < 0: 
            return mx.concat([toks[:to_add-1], mx.array([self.special_tokens["stop"]])])
        else:
            return mx.concat([toks, mx.array([self.special_tokens["pad"]] * to_add)])

    def decode(self, tokens: mx.array, skip_special: bool = True) -> str:
        ids = tokens.tolist()
        if skip_special:
            # <unk> is deliberately kept: it marks content that was lost on
            # encode, so hiding it would make the output look clean but wrong.
            dropped = {self.special_tokens[k] for k in ("start", "stop", "pad")}
            ids = [i for i in ids if i not in dropped]
        return "".join(self.ids_to_tok[i] for i in ids)
