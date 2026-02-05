#!/usr/bin/env python3
from argparse import ArgumentError
import json
import sys
import torch
from pathlib import Path

from dataset_retro import ChunkedSequenceDataset, ShardedChunkedSequenceDataset
from modeling_retro import RetroConfig
from sentence_transformers import SentenceTransformer
from retrieval import RetrieverWithCache, IndexServiceRetriever, IndexServiceClient
from train_retro import RetroModelLMHeadLightning
from data.tokenize_and_chunk import get_tokenizer


class DemoRetriever(RetrieverWithCache):
    """
    Original interactive retriever (fallback if --index-spec is not provided).
    """
    def __init__(self, num_neighbours: int, neighbour_len: int, tokenizer):
        super().__init__(num_neighbours, neighbour_len)
        self.tokenizer = tokenizer

    def get_neighbours_for_chunk(self, chunk: torch.LongTensor):
        """
        chunk - [chunk size]
        Returns:
         neighbours - [num neighbours, neighbour len]
        """
        print("Please input neighbours for the following chunk:")
        print(self.tokenizer.decode(chunk, skip_special_tokens=True))
        ret = torch.zeros((self.num_neighbours, self.neighbour_len), dtype=torch.int64)
        for neighbour_idx in range(self.num_neighbours):
            neighbour_text = input(f"Neighbour {neighbour_idx}: ")
            encoded_neighbour = self.tokenizer.encode(
                neighbour_text,
                return_tensors="pt",
                add_special_tokens=False
            )[0, :self.neighbour_len]
            ret[neighbour_idx, :encoded_neighbour.shape[0]] = encoded_neighbour
        print()
        return ret


def main(args):
    # Load config + tokenizer
    config = RetroConfig(**json.load(args.retro_config.open()))
    tokenizer = get_tokenizer()

    # Choose retriever: real index service or fallback demo
    if args.index_spec is None:
        # Fallback: manual neighbours in terminal (original behavior)
        retriever = DemoRetriever(
            num_neighbours=args.num_neighbours,
            neighbour_len=config.chunk_size * (1 + args.num_continuation_chunks),
            tokenizer=tokenizer
        )
    else:
        # Build retrieval dataset from spec
        spec_json = json.load(args.index_spec.open())
        shards = spec_json if isinstance(spec_json, list) else spec_json.get("shards", [])
        if not shards:
            raise ArgumentError("No shards found in --index-spec")

        base = args.index_spec.parent
        retrieval_dataset = ShardedChunkedSequenceDataset([
            ChunkedSequenceDataset(
                chunks=base / s["chunks"],
                seq2chunk=base / s["seq2chunk"],
                chunk2seq=base / s["chunk2seq"],
            )
            for s in shards
        ])

        # Index service (HTTP) + retrieval model (must match the embedder used for the index)
        client = IndexServiceClient(args.index_host, args.index_port)
        if not client.is_available():
            print(f"[ERROR] Index service not reachable at http://{args.index_host}:{args.index_port}/health", file=sys.stderr)
            sys.exit(1)

        retrieval_model = SentenceTransformer(args.retriever_model_name)
        retriever = IndexServiceRetriever(
            index_service=client,
            retrieval_dataset=retrieval_dataset,
            retrieval_model=retrieval_model,
            tokenizer=tokenizer,
            num_neighbours=args.num_neighbours,
            chunk_size=config.chunk_size,
            num_continuation_chunks=args.num_continuation_chunks,
            verbose=args.retrieval_verbose,
        )

    # Load Lightning checkpoint with retriever injected
    model = RetroModelLMHeadLightning.load_from_checkpoint(
        str(args.checkpoint),
        config=config,
        retriever=retriever
    ).eval()

    # Ensure inputs go to same device as base model parameters
    device = next(model.base.parameters()).device

    prompt = args.prompt
    while True:
        if prompt is None:
            print("Input prompt:")
            prompt = input().strip()

        input_ids = tokenizer(
            [prompt],
            add_special_tokens=False,
            return_tensors="pt"
        )["input_ids"].to(device)

        # IMPORTANT: call generate on the BASE model (has GenerationMixin)
        res = model.base.generate(
            inputs=input_ids,
            do_sample=False,
            num_beams=1,
            top_k=5,
            top_p=1.0,
            temperature=1.0,
            min_length=10,
            max_length=200,
            length_penalty=1.0,
            early_stopping=False,
            num_beam_groups=1,
            num_return_sequences=1,
            repetition_penalty=1.0,
            no_repeat_ngram_size=3,
            encoder_no_repeat_ngram_size=0,
            diversity_penalty=0.0,
            remove_invalid_values=False,
            pad_token_id=config.pad_token_idx,  # use model config pad id
            eos_token_id=1,
            output_scores=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict_in_generate=False,
        )
        prompt = None

        print("\n-- Generation complete --\n")
        print(tokenizer.decode(res[0], skip_special_tokens=True))
        print("\n-------------------------\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--retro-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt")

    parser.add_argument("--num-neighbours", type=int, default=2)
    parser.add_argument("--num-continuation-chunks", type=int, default=1)

    # Retrieval/index wiring (optional; if omitted, falls back to DemoRetriever)
    parser.add_argument("--index-spec", type=Path, help="Path to index spec JSON (same used when building index)")
    parser.add_argument("--index-host", default="localhost")
    parser.add_argument("--index-port", type=int, default=8000)
    parser.add_argument("--retriever-model-name", default="all-MiniLM-L6-v2")
    parser.add_argument("--retrieval-verbose", action="store_true")

    args = parser.parse_args()
    main(args)

