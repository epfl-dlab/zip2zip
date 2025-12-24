from importlib import metadata
import os
from typing import Dict, Any

from regex import F
from utils import setup_seed, save_lm_eval_results_to_yaml, nanoid
from model import OnlineZZModel
from argparse import ArgumentParser
from lm_eval.utils import make_table
from lm_eval.evaluator import simple_evaluate
from utils import PLATFORM_BEST_DTYPE

LM_EVAL_OUT_DIR = "lm_eval_out"


def evaluate(
    model: OnlineZZModel, tasks: str, limit: int, save_samples: bool = False
) -> Dict[str, Any]:
    session_id = nanoid()
    results = simple_evaluate(
        model,
        tasks=tasks.split(","),
        limit=limit,
        num_fewshot=2,
        apply_chat_template=True,
        fewshot_as_multiturn=True,
        confirm_run_unsafe_code=True,
    )

    if save_samples:
        session_dir = os.path.join(LM_EVAL_OUT_DIR, session_id)
        os.makedirs(session_dir, exist_ok=False)
        samples = results["samples"]
        metadata = {k: v for k, v in results.items() if k != "samples"}

        for task_name, task_results in samples.items():
            save_lm_eval_results_to_yaml(
                results=task_results,
                filepath=os.path.join(session_dir, f"{task_name}.yaml"),
            )

        # save the metadata
        save_lm_eval_results_to_yaml(
            results=metadata,
            filepath=os.path.join(session_dir, f"metadata.yaml"),
        )

    return results


if __name__ == "__main__":
    setup_seed()

    parser = ArgumentParser()
    parser.add_argument("--adapter", type=str, required=False)
    parser.add_argument("--hub-adapter", type=str, required=False)
    parser.add_argument("--tasks", type=str, required=False)
    parser.add_argument("--limit", type=int, required=False)
    parser.add_argument(
        "--max-context-length",
        type=int,
        required=False,
        help="This can be used to adjust the context length used in perplexity evaluation.",
    )

    args = parser.parse_args()

    model = OnlineZZModel.load_pretrained(args.adapter, args.hub_adapter).to(
        dtype=PLATFORM_BEST_DTYPE
    )

    if args.max_context_length is not None:
        model.config.seq_length = args.max_context_length

    # model.torch_compile()

    results = evaluate(model, args.tasks, args.limit)

    if results is not None:
        print(make_table(results))
