import os
from typing import Any, Iterable, Iterator
import regex as re
from multiprocessing import Pool


class BPETokenizer:
    """A lightweight BPE Tokenizer for inference."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        vocab:          dict[int, bytes] A dictionary mapping token IDs to their byte representations.
        merges:         list[tuple[bytes, bytes]] A list of BPE merges, ordered by priority.
        special_tokens: list[str] | None = None A list of special tokens to be treated as whole units.
        """

        self.vocab = vocab
        self.merges = merges
        # sort special tokens by length in descending order
        self.special_tokens = (
            sorted(special_tokens, key=len, reverse=True) if special_tokens else []
        )

        self.byte_string_to_id = {v: k for k, v in self.vocab.items()}
        self.merge_ranks = {
            pair: i for i, pair in enumerate(self.merges)
        }  # { pair : rank (position in the dict) }

        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.split_special_token = (
            special_tokens[0].encode("utf-8") if special_tokens else b"\n"
        )
        if special_tokens:
            self.split_special_pattern = (
                f"({ '|'.join([re.escape(st) for st in special_tokens]) })"
            )
        else:
            self.split_special_pattern = None

    # --- Main Functions ---

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ):
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and
        list of merges and a list of special tokens.
        """

        import json

        with open(vocab_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)  # {token_str: token_id}
            vocab = {
                int(token_id): token_str.encode("utf-8")
                for token_str, token_id in data.items()
            }

        # Ensure all 256 bytes are in vocabulary (add missing ones)
        existing_bytes = set(vocab.values())
        max_id = max(vocab.keys()) if vocab else -1
        for byte_val in range(256):
            byte_token = bytes([byte_val])
            if byte_token not in existing_bytes:
                max_id += 1
                vocab[max_id] = byte_token

        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                parts = line.strip().split()
                if len(parts) == 2:
                    p1 = parts[0].encode("utf-8")
                    p2 = parts[1].encode("utf-8")
                    merges.append((p1, p2))

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """

        ids = []
        text_parts = self._split_with_special_tokens(text)
        for part in text_parts:
            if not part:
                continue

            if part in self.special_tokens:
                token_bytes = part.encode("utf-8")
                if token_bytes in self.byte_string_to_id:
                    token_id = self.byte_string_to_id[token_bytes]
                    ids.append(token_id)
            else:
                for match in re.finditer(self.PAT, part):
                    token_str = match.group()
                    try:
                        # tokens = self._apply_bpe(token_str.encode("utf-8"))
                        token_bytes = token_str.encode("utf-8")
                        tokens = self._apply_bpe(token_bytes)
                        token_ids = [self.byte_string_to_id[b] for b in tokens]
                        ids.extend(token_ids)
                    except Exception as e:
                        print(f"Error applying BPE to token: {token_str}")
                        print(f"Error: {e}")
                        raise e
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text
        """
        bytes_list = [self.vocab[id] for id in ids]
        return b"".join(bytes_list).decode("utf-8", errors="replace")

    # --- Helper Functions ---

    @staticmethod
    def get_pairs(parts: list[bytes]) -> set[tuple[bytes, bytes]]:
        """
        Return a set of all adjacent pairs from a list of byte parts.
        """

        return set(zip(parts, parts[1:]))

    def _apply_bpe(self, token_bytes: bytes) -> list[bytes]:
        """
        Apply BPE merges to a bytes sequence using efficient algorithm
        """

        parts = [bytes([b]) for b in token_bytes]
        if len(parts) < 2:
            return parts

        while True:
            # Find the best pair to merge (lowest rank)
            best_pair = None
            best_rank = float("inf")
            best_pos = -1

            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                rank = self.merge_ranks.get(pair, float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_pair = pair
                    best_pos = i

            if best_pair is None or best_rank == float("inf"):
                break

            # Merge the best pair at the found position
            left, right = best_pair
            parts[best_pos] = left + right
            parts.pop(best_pos + 1)

        return parts

    def _split_with_special_tokens(self, text: str) -> list[str]:
        """
        Split text while preserving special tokens, with proper handling of overlapping tokens.
        """
        if not self.special_tokens:
            return [text]

        parts = []
        i = 0
        while i < len(text):
            matched_token = None
            matched_length = 0

            for token in self.special_tokens:
                if text[i:].startswith(token):
                    matched_token = token
                    matched_length = len(token)
                    break

            if matched_token:
                if parts and not parts[-1]:
                    parts.pop()
                parts.append(matched_token)
                i += matched_length
            else:
                if not parts or parts[-1] in self.special_tokens:
                    parts.append("")
                parts[-1] += text[i]
                i += 1

        return [part for part in parts if part]
