import os
from typing import Any, BinaryIO
import collections
import regex as re
from multiprocessing import Pool
import heapq


class _Node:
    def __init__(self, token_id: int, count_ref: dict[str, int]):
        self.id = token_id
        self.count_ref = count_ref  # point to a shared counter dict {'count': ...}
        self.prev: "_Node | None" = None
        self.next: "_Node | None" = None

    @property
    def count(self) -> int:
        return self.count_ref["count"]


class _PriorityQueueItem:
    """
    Represents an item in the priority queue used during BPE merging.
    """

    def __init__(
        self, count: int, p1_bytes: bytes, p2_bytes: bytes, pair: tuple[int, int]
    ):
        self.count = count
        self.p1_bytes = p1_bytes
        self.p2_bytes = p2_bytes
        self.pair = pair

    def __lt__(self, other: "_PriorityQueueItem") -> bool:
        # 1. Higher count has higher priority.
        if self.count != other.count:
            return self.count > other.count
        # 2. If counts are equal, compare p1_bytes lexicographically (larger is higher priority).
        if self.p1_bytes != other.p1_bytes:
            return self.p1_bytes > other.p1_bytes
        # 3. If still equal, compare p2_bytes lexicographically.
        return self.p2_bytes > other.p2_bytes


class BPETrainer:
    """A BPE Trainer to learn vocabulary and merges from a corpus."""

    def __init__(self, vocab_size: int, special_tokens: list[str]):
        self.vocab: dict[int, bytes] = {}
        self.merges: list[tuple[bytes, bytes]] = []
        self.special_tokens = special_tokens
        self.next_id = 0
        self.vocab_size = vocab_size

        # Store special token IDs to prevent them from being merged
        self.special_token_ids = set()

        # one-to-one mapping from special_tokens, bytestring token to ID
        for st in special_tokens:
            self.vocab[self.next_id] = st.encode("utf-8")
            self.special_token_ids.add(self.next_id)  # Mark as special token
            self.next_id += 1
        for i in range(256):
            self.vocab[self.next_id] = bytes([i])
            self.next_id += 1

        self.byte_string_to_id = {v: k for k, v in self.vocab.items()}
        self.pretoken_counts: dict[tuple[bytes, ...], int] = collections.defaultdict(
            int
        )  # { token sequence : count }

        # pattern for pre-tokenization
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # This token is used for finding optimal chunk boundaries for multiprocessing.
        self.split_special_token = (
            special_tokens[0].encode("utf-8") if special_tokens else b"\n"
        )

        # pattern for splitting special tokens
        if special_tokens:
            self.split_special_pattern = (
                f"({ '|'.join([re.escape(st) for st in special_tokens]) })"
            )
        else:
            self.split_special_pattern = None

    # --- Main Functions ---

    def pretokenize(self, input_path: str | os.PathLike):
        with open(input_path, "rb") as f:
            num_processes = 8
            boundaries = self.find_chunk_boundaries(
                f, num_processes, self.split_special_token
            )

            # Prepare arguments for multiprocessing
            chunk_args = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                chunk_args.append(
                    (
                        start,
                        end,
                        input_path,
                        self.PAT,
                        self.special_tokens,
                        self.byte_string_to_id,
                        self.split_special_pattern,
                    )
                )

            with Pool(processes=num_processes) as pool:
                chunk_results = pool.map(self.process_chunk, chunk_args)

            # Merge results from all chunks
            for chunk_pretoken_counts in chunk_results:
                for token_ids, count in chunk_pretoken_counts.items():
                    self.pretoken_counts[token_ids] += count

    def merge(self):
        """
        More efficient implementation of merge.
        Only iterate pretoken_counts once to build the linked list of nodes,
        build a linked list of nodes, as well as the frequency (pair_counts), and index (pair_to_nodes).
        """
        pair_counts: collections.defaultdict[tuple[int, int], int] = (
            collections.defaultdict(int)
        )
        pair_to_nodes: collections.defaultdict[tuple[int, int], set[_Node]] = (
            collections.defaultdict(set)
        )

        for token_id_list, count in self.pretoken_counts.items():
            if len(token_id_list) < 2:
                continue
            count_ref = {"count": count}
            head = _Node(token_id_list[0], count_ref)
            prev_node = head
            for i in range(1, len(token_id_list)):
                current_node = _Node(token_id_list[i], count_ref)
                prev_node.next = current_node
                current_node.prev = prev_node
                pair = (prev_node.id, current_node.id)

                # Skip pairs involving special tokens
                if (
                    prev_node.id not in self.special_token_ids
                    and current_node.id not in self.special_token_ids
                ):
                    pair_counts[pair] += count
                    pair_to_nodes[pair].add(prev_node)

                prev_node = current_node

        pq = []
        for pair, count in pair_counts.items():
            p1_bytes, p2_bytes = self.vocab[pair[0]], self.vocab[pair[1]]
            item = _PriorityQueueItem(
                count, p1_bytes, p2_bytes, pair
            )  # custom comparator
            heapq.heappush(pq, item)

        def update_stats(pair_to_update, delta, node_to_index):
            if not pair_to_update:
                return
            # Skip pairs involving special tokens
            if (
                pair_to_update[0] in self.special_token_ids
                or pair_to_update[1] in self.special_token_ids
            ):
                return

            if delta > 0:
                pair_to_nodes[pair_to_update].add(node_to_index)
            else:
                pair_to_nodes[pair_to_update].discard(node_to_index)
            pair_counts[pair_to_update] += delta

            # push into priority queue
            if pair_counts[pair_to_update] > 0:
                p1_b, p2_b = (
                    self.vocab[pair_to_update[0]],
                    self.vocab[pair_to_update[1]],
                )
                new_item = _PriorityQueueItem(
                    pair_counts[pair_to_update], p1_b, p2_b, pair_to_update
                )
                heapq.heappush(pq, new_item)

        # merges
        num_merges_to_do = self.vocab_size - len(self.vocab)
        for _ in range(num_merges_to_do):
            best_pair = None
            while pq:
                item = heapq.heappop(pq)
                # lazy delete: check if frequency still matches
                if pair_counts.get(item.pair, 0) == item.count:
                    best_pair = item.pair
                    break

            if best_pair is None:
                break
            id1, id2 = best_pair

            new_id = self.allocate_id()
            token1_bytes, token2_bytes = self.vocab[id1], self.vocab[id2]
            new_token_bytes = token1_bytes + token2_bytes
            self.vocab[new_id] = new_token_bytes
            self.byte_string_to_id[new_token_bytes] = new_id
            self.merges.append((token1_bytes, token2_bytes))

            # update the linked list
            valid_pairs = []
            for node1 in pair_to_nodes[best_pair]:
                node2 = node1.next
                # pre-determine the valid pairs, avoid node conflict
                if node2 is not None and node1.id == id1 and node2.id == id2:
                    valid_pairs.append((node1, node2))

            for node1, node2 in valid_pairs:
                word_count = node1.count
                if node1.prev:
                    left_node = node1.prev
                    update_stats(
                        (left_node.id, node1.id), -word_count, left_node
                    )  # update left node
                    update_stats(
                        (left_node.id, new_id), word_count, left_node
                    )  # update new node
                if node2.next:
                    right_node = node2.next
                    update_stats(
                        (node2.id, right_node.id), -word_count, node2
                    )  # update right node
                    update_stats(
                        (new_id, right_node.id), word_count, node1
                    )  # update new node

                node1.id = new_id
                node1.next = node2.next
                if node2.next:
                    node2.next.prev = node1
            del pair_counts[best_pair]
            del pair_to_nodes[best_pair]

    def run_train_bpe(
        self, input_path: str | os.PathLike
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        self.pretokenize(input_path)
        self.merge()
        return self.vocab, self.merges

    def save(self, output_base_path: str):
        """
        Saves the trained vocabulary and merges to files.
        The output filenames are derived from the output_base_path and vocab_size.
        """

        # save vocab
        vocab_filepath = f"{output_base_path}-vocab_size_{self.vocab_size}-vocab.json"
        inverted_vocab = {
            v.decode("utf-8", errors="replace"): k for k, v in self.vocab.items()
        }

        import json

        with open(vocab_filepath, "w", encoding="utf-8") as f:
            json.dump(inverted_vocab, f, ensure_ascii=False, indent=2)
        print(f"Vocabulary saved to: {vocab_filepath}")

        # save merges
        merges_filepath = f"{output_base_path}-vocab_size_{self.vocab_size}-merges.txt"
        with open(merges_filepath, "w", encoding="utf-8") as f:
            for p1, p2 in self.merges:
                # Decode bytes to string for writing
                f.write(
                    f"{p1.decode('utf-8', errors='replace')} {p2.decode('utf-8', errors='replace')}\n"
                )
        print(f"Merges saved to: {merges_filepath}")

    # --- Helper Functions ---

    def allocate_id(self):
        new_id = self.next_id
        self.next_id += 1
        return new_id

    def find_chunk_boundaries(
        self, file: BinaryIO, desired_num_chunks: int, split_special_token: bytes
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """

        assert isinstance(
            split_special_token, bytes
        ), "Must represent special token as a bytestring"
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

    def process_chunk(self, args):
        """Worker function to process a single chunk"""
        (
            start,
            end,
            input_path,
            PAT,
            special_tokens,
            byte_string_to_id,
            split_special_pattern,
        ) = args

        chunk_pretoken_counts = collections.defaultdict(int)
        buffer_size = 10 * 1024 * 1024

        with open(input_path, "rb") as f:
            f.seek(start)
            bytes_to_process = end - start

            while bytes_to_process > 0:
                read_size = min(buffer_size, bytes_to_process)
                chunk_bytes = f.read(read_size)

                if not chunk_bytes:
                    break

                chunk_str = chunk_bytes.decode("utf-8", errors="ignore")
                text_parts = (
                    re.split(split_special_pattern, chunk_str)
                    if split_special_pattern
                    else [chunk_str]
                )

                for part in text_parts:
                    if not part:
                        continue
                    if part in special_tokens:
                        # This part is a special token. It's treated as an atomic unit.
                        token_bytes = part.encode("utf-8")
                        if token_bytes in byte_string_to_id:
                            token_id = byte_string_to_id[token_bytes]
                            # The special token forms a sequence of its own, with length 1.
                            chunk_pretoken_counts[(token_id,)] += 1
                    else:
                        # This part is a regular text segment. Apply the base pre-tokenization regex.
                        for match in re.finditer(PAT, part):
                            token_str = match.group()
                            # Convert the pre-tokenized string into a sequence of byte-level IDs.
                            current_token_ids = [
                                byte_string_to_id[bytes([b_val])]
                                for b_val in token_str.encode("utf-8")
                            ]
                            if current_token_ids:
                                chunk_pretoken_counts[tuple(current_token_ids)] += 1

                bytes_to_process -= read_size

        return chunk_pretoken_counts
