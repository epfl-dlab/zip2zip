#!/usr/bin/env python3
"""
Visualize d distribution for zip2zip hypertokens.

Definition:
- V0: initial vocabulary size
- For hypertoken id t used at position pos (0-based):
    create_pos = t - V0
    d = pos - create_pos
"""

# Run:
# python research/visualize_d_distribution.py --model epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from zip2zip.tokenizer import Zip2ZipTokenizer


DOMAIN_TO_DATASET = {
    "web": "HuggingFaceFW/fineweb-edu",
    "code": "devngho/the-stack-llm-annotations-v2",
    "math": "AI-MO/NuminaMath-1.5",
    "chat": "HuggingFaceH4/ultrachat_200k",
}

DOMAIN_SPLIT_CANDIDATES = ["train", "train_sft", "train_gen"]


def ensure_batched_input_ids(input_ids):
    if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
        return input_ids
    return [input_ids]


def first_nonempty_string(obj):
    if isinstance(obj, str):
        s = obj.strip()
        return s if s else None
    if isinstance(obj, dict):
        for v in obj.values():
            found = first_nonempty_string(v)
            if found:
                return found
    if isinstance(obj, list):
        for v in obj:
            found = first_nonempty_string(v)
            if found:
                return found
    return None


def extract_text_from_example(domain, example):
    if domain in ("web", "code"):
        for key in ("text", "content", "code", "document", "body"):
            value = example.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    if domain == "math":
        pieces = []
        for key in ("problem", "question", "prompt", "solution", "answer", "text"):
            value = example.get(key)
            if isinstance(value, str) and value.strip():
                pieces.append(value.strip())
        if pieces:
            return "\n\n".join(pieces)

    if domain == "chat":
        for key in ("messages", "conversation", "conversations"):
            conv = example.get(key)
            if not isinstance(conv, list):
                continue
            pieces = []
            for turn in conv:
                if isinstance(turn, dict):
                    content = turn.get("content")
                    if isinstance(content, str) and content.strip():
                        pieces.append(content.strip())
                elif isinstance(turn, str) and turn.strip():
                    pieces.append(turn.strip())
            if pieces:
                return "\n".join(pieces)

    return first_nonempty_string(example)


def sample_domain_texts(domain, max_samples, min_chars, streaming, seed):
    dataset_name = DOMAIN_TO_DATASET[domain]
    last_error = None
    dataset = None
    used_split = None
    for split in DOMAIN_SPLIT_CANDIDATES:
        try:
            dataset = load_dataset(dataset_name, split=split, streaming=streaming)
            used_split = split
            break
        except Exception as exc:
            last_error = exc
    if dataset is None:
        raise RuntimeError(
            f"failed loading {dataset_name} with splits={DOMAIN_SPLIT_CANDIDATES}: {last_error}"
        )

    if streaming:
        try:
            dataset = dataset.shuffle(
                seed=seed, buffer_size=min(10_000, max(2_000, max_samples * 20))
            )
        except Exception:
            pass

    texts = []
    for example in dataset:
        text = extract_text_from_example(domain, example)
        if not text or len(text) < min_chars:
            continue
        texts.append(text)
        if len(texts) >= max_samples:
            break

    return texts, used_split


def load_input_texts(args):
    if args.use_domains:
        domain_to_texts = {}
        for domain in args.domains:
            texts, split_name = sample_domain_texts(
                domain=domain,
                max_samples=args.max_samples_per_domain,
                min_chars=args.min_chars,
                streaming=args.streaming,
                seed=args.seed,
            )
            print(f"[{domain}] loaded {len(texts)} samples (split={split_name})")
            domain_to_texts[domain] = texts
        return domain_to_texts

    texts = []
    if args.text:
        texts.extend(args.text)
    if args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as file:
            texts.extend([line.strip() for line in file if line.strip()])
    if not texts:
        raise ValueError("no input text provided; use --text/--text-file or --use-domains")
    return {"input": texts}


def compute_d_values(tokenizer, text, v0, max_seq_len):
    tokenize_kwargs = {"return_codebook": True, "verbose": False}
    if max_seq_len > 0:
        tokenize_kwargs["truncation"] = True
        tokenize_kwargs["max_length"] = max_seq_len
    enc = tokenizer(text, **tokenize_kwargs)

    d_values = []
    for ids, codebook in zip(ensure_batched_input_ids(enc["input_ids"]), enc["codebooks"]):
        hyper_ids = {int(key) for key in codebook.to_dict().keys()}
        for pos, token_id in enumerate(ids):
            token_id = int(token_id)
            if token_id not in hyper_ids:
                continue
            create_pos = token_id - v0
            d_values.append(pos - create_pos)
    return d_values


def summarize(values, num_samples, v0):
    arr = np.array(values, dtype=np.float64)
    return {
        "num_samples": int(num_samples),
        "num_hypertoken_occurrences": int(len(arr)),
        "vocab0": int(v0),
        "d_min": float(arr.min()),
        "d_p25": float(np.percentile(arr, 25)),
        "d_median": float(np.median(arr)),
        "d_mean": float(arr.mean()),
        "d_p75": float(np.percentile(arr, 75)),
        "d_max": float(arr.max()),
        "d_std": float(arr.std()),
    }


def save_figures(domain_to_d, bins, out_dir):
    nonempty = {k: v for k, v in domain_to_d.items() if v}
    if not nonempty:
        raise ValueError("no hypertoken occurrences found")

    out_dir.mkdir(parents=True, exist_ok=True)
    hist_path = out_dir / "d_log2_1pd_hist.png"
    ecdf_path = out_dir / "d_log2_1pd_ecdf.png"

    plt.figure(figsize=(10, 6))
    for domain, values in nonempty.items():
        log_values = np.log2(np.maximum(np.array(values, dtype=np.float64), 0.0) + 1.0)
        plt.hist(log_values, bins=bins, alpha=0.35, density=True, label=f"{domain} (n={len(values)})")
    plt.title("Histogram of log2(1 + d)")
    plt.xlabel("log2(1 + d)")
    plt.ylabel("density")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(hist_path, dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    for domain, values in nonempty.items():
        log_values = np.log2(np.maximum(np.array(values, dtype=np.float64), 0.0) + 1.0)
        x = np.sort(log_values)
        y = np.arange(1, len(x) + 1) / len(x)
        plt.plot(x, y, linewidth=2, label=f"{domain} (n={len(values)})")
    plt.title("ECDF of log2(1 + d)")
    plt.xlabel("log2(1 + d)")
    plt.ylabel("ECDF")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ecdf_path, dpi=160)
    plt.close()

    return hist_path, ecdf_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="zip2zip model id/path")
    parser.add_argument(
        "--use-domains",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use predefined HF corpora instead of --text/--text-file.",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["web", "code", "math", "chat"],
        choices=["web", "code", "math", "chat"],
    )
    parser.add_argument("--max-samples-per-domain", type=int, default=400)
    parser.add_argument("--min-chars", type=int, default=32)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text", action="append", default=None)
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--out-dir", default="research/outputs/visualizations/d")
    parser.add_argument("--save-json", action="store_true", help="Also save JSON stats.")
    return parser.parse_args()


def main():
    args = parse_args()
    domain_to_texts = load_input_texts(args)
    tokenizer = Zip2ZipTokenizer.from_pretrained(args.model)
    v0 = int(tokenizer.initial_vocab_size)

    domain_to_d = {}
    for domain, texts in domain_to_texts.items():
        values = []
        for text in texts:
            values.extend(compute_d_values(tokenizer, text, v0, args.max_seq_len))
        domain_to_d[domain] = values

    hist_path, ecdf_path = save_figures(domain_to_d, args.bins, Path(args.out_dir))
    print(f"Saved: {hist_path}")
    print(f"Saved: {ecdf_path}")

    if args.save_json:
        nonempty = {k: v for k, v in domain_to_d.items() if v}
        all_values = [v for values in nonempty.values() for v in values]
        payload = {
            "overall_stats": summarize(
                all_values,
                num_samples=sum(len(v) for v in domain_to_texts.values()),
                v0=v0,
            ),
            "per_domain_stats": {
                domain: summarize(values, len(domain_to_texts[domain]), v0)
                for domain, values in nonempty.items()
            },
        }
        json_path = Path(args.out_dir) / "d_stats.json"
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=True, indent=2)
        print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
