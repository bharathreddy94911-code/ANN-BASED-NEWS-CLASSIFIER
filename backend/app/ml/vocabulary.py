import json
from collections import Counter
from typing import List, Dict, Optional, Tuple, Iterable

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
PAD_IDX = 0
UNK_IDX = 1

class Vocabulary:
    def __init__(self, min_freq: int = 2, max_size: Optional[int] = 25000):
        self.min_freq = min_freq
        self.max_size = max_size
        self.w2i: Dict[str, int] = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self.i2w: Dict[int, str] = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN}
        self.word_counts: Dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.w2i)

    def build_from_tokens(self, tokenized_texts: Iterable[List[str]]):
        counter = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)
        
        self.word_counts = dict(counter)
        
        # Filter by min_freq and sort by frequency
        sorted_words = [w for w, c in counter.most_common() if c >= self.min_freq]
        if self.max_size is not None:
            sorted_words = sorted_words[: self.max_size - 2] # reserve space for special tokens
            
        for idx, word in enumerate(sorted_words, start=2):
            self.w2i[word] = idx
            self.i2w[idx] = word

    def encode(self, tokens: List[str], max_len: Optional[int] = 64) -> List[int]:
        ids = [self.w2i.get(token, UNK_IDX) for token in tokens]
        if max_len is not None:
            if len(ids) > max_len:
                ids = ids[:max_len]
            elif len(ids) < max_len:
                ids = ids + [PAD_IDX] * (max_len - len(ids))
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> List[str]:
        words = []
        for i in ids:
            w = self.i2w.get(i, UNK_TOKEN)
            if skip_special and w in (PAD_TOKEN, UNK_TOKEN):
                continue
            words.append(w)
        return words

    def save(self, filepath: str):
        data = {
            'min_freq': self.min_freq,
            'max_size': self.max_size,
            'w2i': self.w2i,
            'word_counts': self.word_counts
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> 'Vocabulary':
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        vocab = cls(min_freq=data['min_freq'], max_size=data['max_size'])
        vocab.w2i = data['w2i']
        vocab.i2w = {int(v): k for k, v in data['w2i'].items()}
        vocab.word_counts = data.get('word_counts', {})
        return vocab
