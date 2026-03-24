#!/usr/bin/env python3
"""
Visualize hypertoken base-token-length distribution across domains.

For each hypertoken occurrence:
- Expand codebook rule recursively to count base-token length.
- Plot per-domain proportions by expanded length.
"""

# Run:
# python research/visualize_length_distribution.py --model epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1

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
        raise RuntimeError(f"failed loading {dataset_name}: {last_error}")

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


def ensure_batched_input_ids(input_ids):
    if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
        return input_ids
    return [input_ids]


def normalize_rule(rule_value):
    if isinstance(rule_value, int):
        return [rule_value]
    if isinstance(rule_value, list):
        out = []
        for x in rule_value:
            try:
                out.append(int(x))
            except (TypeError, ValueError):
                continue
        return out
    return []


def expanded_length(token_id, rules, memo, visiting):
    if token_id in memo:
        return memo[token_id]
    if token_id not in rules:
        return 1
    if token_id in visiting:
        return 1

    visiting.add(token_id)
    total = 0
    for child in rules[token_id]:
        total += expanded_length(child, rules, memo, visiting)
    visiting.remove(token_id)
    memo[token_id] = max(1, total)
    return memo[token_id]


def compute_length_values(tokenizer, text):
    enc = tokenizer(text, return_codebook=True)
    values = []
    for ids, codebook in zip(ensure_batched_input_ids(enc["input_ids"]), enc["codebooks"]):
        rules = {}
        for key, value in codebook.to_dict().items():
            try:
                token_id = int(key)
            except (TypeError, ValueError):
                continue
            rules[token_id] = normalize_rule(value)
        if not rules:
            continue

        memo = {}
        for token_id in rules:
            expanded_length(token_id, rules, memo, set())

        for tok in ids:
            tok = int(tok)
            if tok in rules:
                values.append(int(memo.get(tok, 1)))
    return values


def summarize(values, num_samples):
    arr = np.array(values, dtype=np.float64)
    return {
        "num_samples": int(num_samples),
        "num_hypertoken_occurrences": int(len(arr)),
        "len_min": float(arr.min()),
        "len_p25": float(np.percentile(arr, 25)),
        "len_median": float(np.median(arr)),
        "len_mean": float(arr.mean()),
        "len_p75": float(np.percentile(arr, 75)),
        "len_max": float(arr.max()),
        "len_std": float(arr.std()),
    }


def build_domain_proportions(domain_to_values, max_length_x):
    labels = [str(i) for i in range(1, max_length_x + 1)]
    proportions = {}
    for domain, values in domain_to_values.items():
        arr = np.array(values, dtype=np.int64)
        counts = np.zeros(len(labels), dtype=np.float64)
        if len(arr) == 0:
            proportions[domain] = counts
            continue
        for v in arr:
            if v <= 0:
                continue
            if v <= max_length_x:
                counts[v - 1] += 1.0
        proportions[domain] = counts / len(arr)
    return labels, proportions


def save_figure(domain_to_values, out_dir):
    nonempty = {k: v for k, v in domain_to_values.items() if v}
    if not nonempty:
        raise ValueError("no hypertoken occurrences found")

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "hyper_base_len_distribution.png"

    max_length_x = max(max(values) for values in nonempty.values())
    labels, domain_props = build_domain_proportions(
        nonempty, max_length_x=max_length_x
    )
    x = np.arange(len(labels))
    width = 0.8 / len(nonempty)

    plt.figure(figsize=(12, 6))
    for i, domain in enumerate(nonempty.keys()):
        offset = (i - (len(nonempty) - 1) / 2.0) * width
        plt.bar(
            x + offset,
            domain_props[domain],
            width=width,
            alpha=0.85,
            label=f"{domain} (n={len(nonempty[domain])})",
        )
    plt.title("Hypertoken Expanded Base-Length Distribution")
    plt.xlabel("expanded base-token length")
    plt.ylabel("proportion within domain")
    plt.xticks(x, labels)
    plt.grid(axis="y", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()
    return fig_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="zip2zip model id/path")
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
    parser.add_argument("--out-dir", default="research/outputs/visualizations/length")
    parser.add_argument("--save-json", action="store_true", help="Also save JSON stats.")
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = Zip2ZipTokenizer.from_pretrained(args.model)

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

    domain_to_values = {}
    for domain, texts in domain_to_texts.items():
        values = []
        for text in texts:
            values.extend(compute_length_values(tokenizer, text))
        domain_to_values[domain] = values

    fig_path = save_figure(domain_to_values, Path(args.out_dir))
    print(f"Saved: {fig_path}")

    if args.save_json:
        nonempty = {k: v for k, v in domain_to_values.items() if v}
        all_values = [v for values in nonempty.values() for v in values]
        payload = {
            "overall_stats": summarize(all_values, num_samples=sum(len(v) for v in domain_to_texts.values())),
            "per_domain_stats": {
                domain: summarize(values, num_samples=len(domain_to_texts[domain]))
                for domain, values in nonempty.items()
            },
        }
        json_path = Path(args.out_dir) / "hyper_base_len_stats.json"
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=True, indent=2)
        print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
