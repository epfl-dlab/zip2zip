from datetime import datetime
import os
import torch
from zip2zip.model import Zip2ZipModel
from zip2zip.tokenizer import Zip2ZipTokenizer
from zip2zip.tools.harness import Zip2ZipForLMEval, save_lm_eval_results_to_yaml
from lm_eval.utils import make_table

if __name__ == "__main__":
    from lm_eval.evaluator import simple_evaluate

    # # model_name = "epfl-dlab/zip2zip-Llama-3.2-3B-Instruct-v0.1"
    # model_name = "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1"
    # model = Zip2ZipModel.from_pretrained(
    #     model_name, device_map="cuda", torch_dtype=torch.float16, max_subtokens=3
    # )
    # tokenizer = Zip2ZipTokenizer.from_pretrained(model_name, max_subtokens=3)

    checkpoint_path = "out/zip2zip-train-FT-M3-L2-S1024/checkpoint-36000"

    model = Zip2ZipModel.from_checkpoint(
        checkpoint_path,
        device_map="auto",
        dtype=torch.float16,
    )

    tokenizer = Zip2ZipTokenizer.from_pretrained(checkpoint_path)

    model = Zip2ZipForLMEval(
        model,
        tokenizer,
        max_length=1024,
        batch_size=1,
    )

    model.model.eval()

    # disable all parameters in the model
    for param in model.model.parameters():
        param.requires_grad = False

    results = simple_evaluate(
        model,
        # tasks="arc_easy",
        # ai2_arc,openbookqa,piqa,winogrande,commonsense_qa,lambada,mathqa,hellaswag
        # tasks=["openbookqa", "piqa", "winogrande", "commonsense_qa", "lambada", "mathqa", "hellaswag"],
        # tasks=["wmt14-fr-en", "wmt14-en-fr"],
        tasks=["paloma_mc4", "paloma_dolma_100_programing_languages"],
        limit=100,
        num_fewshot=2,
        apply_chat_template=True,
        fewshot_as_multiturn=True,
        confirm_run_unsafe_code=True,
    )

    if results is not None:
        print(make_table(results))

    if True:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        session_dir = os.path.join("eleutherai_eval", current_time)
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
