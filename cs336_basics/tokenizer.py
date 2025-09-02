import re
from collections.abc import Iterable, Iterator

from cs336_basics.train_bpe import PRETOKENIZATION_REGEX, load_merges, load_vocab
from cs336_basics.utils import sample_documents


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges

        # Still have to declare what special tokens are present even though they are already in the vocab,
        # since they are treated like just any other token otherwise.
        # Sort in decreasing length to greedily match longest special token in the case of overlapping special tokens.
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else None

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        vocab = load_vocab(vocab_filepath)
        merges = load_merges(merges_filepath)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        bytes_to_id = {v: k for k, v in self.vocab.items()}

        # Handle special tokens by splitting text
        parts = [text]
        if self.special_tokens:
            special_token_pattern = "|".join(re.escape(token) for token in self.special_tokens)
            special_regex = re.compile(f"({special_token_pattern})")
            parts = special_regex.split(text)

        token_ids = []

        for part in parts:
            if not part:
                continue

            # Check if this part is a special token
            if self.special_tokens and part in self.special_tokens:
                special_token_bytes = part.encode("utf-8")
                if special_token_bytes in bytes_to_id:
                    token_ids.append(bytes_to_id[special_token_bytes])
                continue

            # Pretokenize this part
            for match in PRETOKENIZATION_REGEX.finditer(part):
                # Convert match to list of individual bytes
                pretoken = list(match.group().encode("utf-8"))
                tokens = [bytes([b]) for b in pretoken]

                # Apply merges in order
                for merge in self.merges:
                    i = 0
                    while i < len(tokens) - 1:
                        if tokens[i] == merge[0] and tokens[i + 1] == merge[1]:
                            # Merge the pair
                            tokens[i : i + 2] = [merge[0] + merge[1]]
                        else:
                            i += 1

                # Convert tokens to IDs
                for token in tokens:
                    if token in bytes_to_id:
                        token_ids.append(bytes_to_id[token])

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        byte_sequence = b""
        for token_id in ids:
            if token_id in self.vocab:
                byte_sequence += self.vocab[token_id]

        return byte_sequence.decode("utf-8", errors="replace")


if __name__ == "__main__":
    vocab_filepath = "data/TinyStoriesV2-GPT4-train-vocab.txt"
    merges_filepath = "data/TinyStoriesV2-GPT4-train-merges.txt"
    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)

    docs = sample_documents("data/TinyStoriesV2-GPT4-valid.txt", num_samples=10, num_load=100, seed=123)

    total_bytes_count = 0
    total_ids_count = 0
    for doc in docs:
        token_ids = tokenizer.encode(doc)
        total_bytes_count += len(doc.encode("utf-8"))
        total_ids_count += len(token_ids)

    print(f"Total bytes: {total_bytes_count}")
    print(f"Compression ratio (bytes/token): {total_bytes_count / total_ids_count}")
