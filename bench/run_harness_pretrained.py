import argparse
from datetime import datetime
import os
import torch
from zip2zip.model import Zip2ZipModel
from zip2zip.tokenizer import Zip2ZipTokenizer
from zip2zip.tools.harness import Zip2ZipForLMEval, save_lm_eval_results_to_yaml
from lm_eval.utils import make_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str,
                        help="HF repo or local path to zip2zip model")
    parser.add_argument("--tasks", type=str, nargs="+",
                        default=["wikitext", "piqa", "winogrande", "gsm8k"])
    parser.add_argument("--max_subtokens", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_fewshot", type=int, default=2)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output_dir", type=str, default="eleutherai_eval")
    args = parser.parse_args()

    from lm_eval.evaluator import simple_evaluate

    dtype = getattr(torch, args.dtype)
    model = Zip2ZipModel.from_pretrained(
        args.model_path, torch_dtype=dtype, max_subtokens=args.max_subtokens,
    ).to("cuda").eval()

    tokenizer = Zip2ZipTokenizer.from_pretrained(
        args.model_path, max_subtokens=args.max_subtokens,
    )

    model = Zip2ZipForLMEval(
        model, tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    for param in model.model.parameters():
        param.requires_grad = False

    results = simple_evaluate(
        model,
        tasks=args.tasks,
        limit=args.limit,
        num_fewshot=args.num_fewshot,
        confirm_run_unsafe_code=True,
    )

    if results is not None:
        print(make_table(results))

        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        session_dir = os.path.join(args.output_dir, current_time)
        os.makedirs(session_dir, exist_ok=False)
        samples = results["samples"]
        metadata = {k: v for k, v in results.items() if k != "samples"}

        for task_name, task_results in samples.items():
            save_lm_eval_results_to_yaml(
                results=task_results,
                filepath=os.path.join(session_dir, f"{task_name}.yaml"),
            )
        save_lm_eval_results_to_yaml(
            results=metadata,
            filepath=os.path.join(session_dir, "metadata.yaml"),
        )


if __name__ == "__main__":
    main()
