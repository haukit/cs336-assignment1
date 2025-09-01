import argparse
import multiprocessing as mp
import os
from collections import Counter
from pathlib import Path
from typing import BinaryIO

import regex as re

GPT2_PRETOKENIZATION_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PRETOKENIZATION_REGEX = re.compile(GPT2_PRETOKENIZATION_PATTERN)


def initialize_vocabulary(special_tokens: list[str]) -> dict[int, bytes]:
    special_token_bytes = [token.encode("utf-8") for token in special_tokens]
    single_bytes = [bytes([x]) for x in range(256)]
    all_bytes = special_token_bytes + single_bytes

    token_ids = list(range(len(all_bytes)))

    return dict(zip(token_ids, all_bytes))


def pretokenize(text: str) -> Counter:
    pretoken_dict = Counter()
    for match in PRETOKENIZATION_REGEX.finditer(text):
        pretoken = match.group().encode("utf-8")
        pretoken = tuple(bytes([x]) for x in pretoken)
        pretoken_dict[pretoken] += 1

    return pretoken_dict


def pretokenize_chunk(args: tuple[str, str]) -> Counter:
    """Worker function for multiprocessing pretokenization"""
    chunk, special_token_pattern = args

    chunk_pretoken_dict = Counter()

    if special_token_pattern:
        # Find all special tokens and their positions
        special_tokens_in_chunk = []
        for match in re.finditer(special_token_pattern, chunk):
            special_tokens_in_chunk.append((match.start(), match.end(), match.group()))

        # Process text between special tokens; the special tokens themselves do not need to be counted
        last_end = 0
        for start, end, token_text in special_tokens_in_chunk:
            # Process text before this special token
            if start > last_end:
                subchunk = chunk[last_end:start]
                if subchunk:
                    subchunk_pretoken_dict = pretokenize(subchunk)
                    chunk_pretoken_dict.update(subchunk_pretoken_dict)

            last_end = end

        # Process any remaining text after the last special token
        if last_end < len(chunk):
            subchunk = chunk[last_end:]
            if subchunk:
                subchunk_pretoken_dict = pretokenize(subchunk)
                chunk_pretoken_dict.update(subchunk_pretoken_dict)

    else:
        # No special tokens, process normally
        chunk_pretoken_dict = pretokenize(chunk)

    return chunk_pretoken_dict


def find_merge(pretoken_dict: Counter) -> tuple[bytes, bytes]:
    # Find most frequent pairs
    pair_dict = Counter()
    for pretoken, pretoken_count in pretoken_dict.items():
        for i in range(len(pretoken) - 1):
            pair_dict[pretoken[i : i + 2]] += pretoken_count

    if not pair_dict:
        # No pairs to merge
        return None

    max_pair_freq = max(pair_dict.values())
    merge_candidates = [pair for pair, freq in pair_dict.items() if freq == max_pair_freq]

    # Return lexicographically greatest pair
    return max(merge_candidates)


def apply_merge(
    pretoken_dict: Counter,
    merge: tuple[bytes, bytes],
) -> Counter:
    if merge is None:
        return pretoken_dict

    pretokens = list(pretoken_dict.keys())
    merged = b"".join(merge)

    for pretoken in pretokens:
        new_pretoken = []

        i = 0
        while i < len(pretoken):
            if i + 1 < len(pretoken) and pretoken[i] == merge[0] and pretoken[i + 1] == merge[1]:
                new_pretoken.append(merged)
                i += 2
            else:
                new_pretoken.append(pretoken[i])
                i += 1

        new_pretoken = tuple(new_pretoken)

        if pretoken != new_pretoken:
            pretoken_dict[new_pretoken] = pretoken_dict[pretoken]
            del pretoken_dict[pretoken]

    return pretoken_dict


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[list[bytes], list[list[bytes]]]:
    merges = []
    num_processes = os.cpu_count()
    special_token_pattern = "|".join(re.escape(token) for token in special_tokens) if special_tokens else None

    # Initialize vocabulary
    vocab = initialize_vocabulary(special_tokens)
    num_merges = vocab_size - len(vocab)

    # Pretokenize: Split corpus into chunks and prepare inputs for multiprocessing worker function
    chunks = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk_text = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append((chunk_text, special_token_pattern))

    # Pretokenize: Use multiprocessing to pretokenize each chunk and merge results
    if len(chunks) > 1:
        with mp.Pool(processes=num_processes) as pool:
            chunk_pretoken_dicts = pool.map(pretokenize_chunk, chunks)
    else:
        # Fallback to sequential processing
        chunk_pretoken_dicts = [pretokenize_chunk(chunk) for chunk in chunks]

    pretoken_dict = Counter()
    for d in chunk_pretoken_dicts:
        pretoken_dict.update(d)

    print(f"Pretokenization complete. Found {len(pretoken_dict)} unique pretokens")
    print(f"Computing {num_merges} merges...")

    # Compute merges
    merges = []
    for i in range(num_merges):
        if i % 100 == 0:
            print(f"Merge {i}/{num_merges}")

        merge = find_merge(pretoken_dict)
        pretoken_dict = apply_merge(pretoken_dict, merge)

        merges.append(merge)
        vocab[len(vocab)] = b"".join(merge)

    return vocab, merges


def save_vocab(vocab: dict[int, bytes], filepath: str) -> None:
    with open(filepath, "w") as f:
        for token_id in sorted(vocab.keys()):
            f.write(f"{token_id} {repr(vocab[token_id])}\n")


def load_vocab(filepath: str) -> dict[int, bytes]:
    vocab = {}
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                token_id = int(parts[0])
                token_bytes = eval(parts[1])  # Safe because we control the format
                vocab[token_id] = token_bytes
    return vocab


def save_merges(merges: list[tuple[bytes]], filepath: str) -> None:
    with open(filepath, "w") as f:
        for merge_pair in merges:
            f.write(f"{repr(merge_pair[0])} {repr(merge_pair[1])}\n")


def load_merges(filepath: str) -> list[tuple[bytes]]:
    merges = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                first_bytes = eval(parts[0])  # Safe because we control the format
                second_bytes = eval(parts[1])
                merges.append((first_bytes, second_bytes))
    return merges


if __name__ == "__main__":
    # Train BPE
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path, help="Path to input text file")
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        help="Directory to save vocab and merges (default: same as input file)",
    )
    parser.add_argument("--vocab_size", type=int, default=1000, help="Vocabulary size")
    parser.add_argument(
        "--special_tokens",
        nargs="*",
        default=["<|endoftext|>"],
        help="List of special tokens (space separated)",
    )
    args = parser.parse_args()

    vocab, merges = train_bpe(args.input_path, args.vocab_size, args.special_tokens)

    # Default output_dir = input file's parent
    out_dir = args.output_dir or args.input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    base = args.input_path.stem
    vocab_file = out_dir / f"{base}-vocab.txt"
    merges_file = out_dir / f"{base}-merges.txt"

    save_vocab(vocab, vocab_file)
    save_merges(merges, merges_file)
