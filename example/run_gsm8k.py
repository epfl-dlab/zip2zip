import torch
import re
import os, sys
import random
import transformers
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import copy
import os
import ssl
import urllib.request

import os.path as osp
import gzip
import json
from colorama import Fore, Style

import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from interface import load_model, generate

transformers.logging.set_verbosity(40)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 8
COT_FLAG = True
DEBUG = False
ANSWER_TRIGGER = "The answer is"


def print_green(text):
    return f"{Fore.GREEN}{text}{Style.RESET_ALL}"


def print_red(text):
    return f"{Fore.RED}{text}{Style.RESET_ALL}"


def download_url(url: str, folder="folder"):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition("/")[2]
    file = file if file[0] == "?" else file.split("?")[0]
    path = osp.join(folder, file)
    if osp.exists(path):
        print(f"File {file} exists, use existing file.")
        return path

    print(f"Downloading {url}")
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, "wb") as f:
        f.write(data.read())

    return path


def load_jsonl(
    file_path,
    instruction="instruction",
    input="input",
    output="output",
    category="category",
    is_gzip=False,
):
    # Format of each line:
    # {'instruction': ..., 'input': ..., 'output':...}
    list_data_dict = []
    open_func = open if not is_gzip else gzip.open
    with open_func(file_path, "r") as f:
        for line in f:
            item = json.loads(line)
            new_item = dict(
                instruction=item[instruction] if instruction in item else None,
                input=item[input] if input in item else None,
                output=item[output] if output in item else None,
                category=item[category] if category in item else None,
            )
            item = new_item
            list_data_dict.append(item)
    return list_data_dict


def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_answer, answer):
    gt_answer = extract_answer_from_output(answer)
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer


def create_demo_text(n_shot=8, cot_flag=True):
    question, chain, answer = [], [], []
    question.append(
        "There are 15 trees in the grove. "
        "Grove workers will plant trees in the grove today. "
        "After they are done, there will be 21 trees. "
        "How many trees did the grove workers plant today?"
    )
    chain.append(
        "There are 15 trees originally. "
        "Then there were 21 trees after some more were planted. "
        "So there must have been 21 - 15 = 6."
    )
    answer.append("6")

    question.append(
        "If there are 3 cars in the parking lot and 2 more cars arrive, "
        "how many cars are in the parking lot?"
    )
    chain.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
    answer.append("5")

    question.append(
        "Leah had 32 chocolates and her sister had 42. If they ate 35, "
        "how many pieces do they have left in total?"
    )
    chain.append(
        "Originally, Leah had 32 chocolates. "
        "Her sister had 42. So in total they had 32 + 42 = 74. "
        "After eating 35, they had 74 - 35 = 39."
    )
    answer.append("39")

    question.append(
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
        "has 12 lollipops. How many lollipops did Jason give to Denny?"
    )
    chain.append(
        "Jason started with 20 lollipops. Then he had 12 after giving some "
        "to Denny. So he gave Denny 20 - 12 = 8."
    )
    answer.append("8")

    question.append(
        "Shawn has five toys. For Christmas, he got two toys each from his "
        "mom and dad. How many toys does he have now?"
    )
    chain.append(
        "Shawn started with 5 toys. If he got 2 toys each from his mom and "
        "dad, then that is 4 more toys. 5 + 4 = 9."
    )
    answer.append("9")

    question.append(
        "There were nine computers in the server room. Five more computers "
        "were installed each day, from monday to thursday. "
        "How many computers are now in the server room?"
    )
    chain.append(
        "There were originally 9 computers. For each of 4 days, 5 more "
        "computers were added. So 5 * 4 = 20 computers were added. "
        "9 + 20 is 29."
    )
    answer.append("29")

    question.append(
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
        "wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?"
    )
    chain.append(
        "Michael started with 58 golf balls. After losing 23 on tuesday, "
        "he had 58 - 23 = 35. After losing 2 more, "
        "he had 35 - 2 = 33 golf balls."
    )
    answer.append("33")

    question.append(
        "Olivia has $23. She bought five bagels for $3 each. "
        "How much money does she have left?"
    )
    chain.append(
        "Olivia had 23 dollars. "
        "5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
        "So she has 23 - 15 dollars left. 23 - 15 is 8."
    )
    answer.append("8")

    # randomize order of the examples ...
    index_list = list(range(len(question)))
    random.shuffle(index_list)

    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list[:n_shot]:
        if cot_flag:
            demo_text += (
                "Q: "
                + question[i]
                + "\nA: "
                + chain[i]
                + " "
                + ANSWER_TRIGGER
                + " "
                + answer[i]
                + ".\n\n"
            )
        else:
            demo_text += (
                "Question: "
                + question[i]
                + "\nAnswer: "
                + ANSWER_TRIGGER
                + " "
                + answer[i]
                + ".\n\n"
            )
    return demo_text


def build_prompt(input_text, n_shot, cot_flag):
    demo = create_demo_text(n_shot, cot_flag)
    input_text_prompt = demo + "Q: " + input_text + "\n" + "A:"
    return input_text_prompt


def clean_answer(model_pred):
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = True if len(preds) > 1 else False
    if answer_flag:
        # Pick first answer with flag
        pred = preds[1]
    else:
        # Pick last number without flag
        pred = preds[-1]

    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]

    if len(pred) == 0:
        return INVALID_ANS

    if answer_flag:
        # choose the first element in list
        pred = pred[0]
    else:
        # choose the last element in list
        pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred[-1] == ".":
        pred = pred[:-1]

    return pred


def seed_everything(seed: int):
    import random
    import os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=False,
        help="The model checkpoint for weights initialization.",
    )
    parser.add_argument("--adapter", type=str, required=False)
    parser.add_argument("--hub-adapter", type=str, required=False)

    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="The root folder of the data.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./local_gsm8k_output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--n_shot",
        type=int,
        default=8,
        help="Number of demonstration examples.",
    )
    parser.add_argument(
        "--num_instances",
        type=int,
        default=50,
        help="Number of test instances to evaluate.",
    )
    parser.add_argument(
        "--disable_kv_cache",
        action="store_true",
        help="Disable key-value cache.",
    )

    parser.add_argument("--load", type=str, default=None, help="load quantized model")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    seed_everything(args.seed)

    test_filepath = os.path.join(args.data_root, "gsm8k_test.jsonl")
    if not os.path.exists(test_filepath):
        download_url(
            "https://raw.githubusercontent.com/openai/"
            "grade-school-math/2909d34ef28520753df82a2234c357259d254aa8/"
            "grade_school_math/data/test.jsonl",
            args.data_root,
        )
        os.rename(os.path.join(args.data_root, "test.jsonl"), test_filepath)

    list_data_dict = load_jsonl(test_filepath, instruction="question", output="answer")
    # Limit the number of instances to evaluate
    list_data_dict = list_data_dict[: args.num_instances]

    model, tokenizer = load_model(
        args.model_name_or_path, args.adapter, args.hub_adapter
    )

    if args.load:
        print("loading...", args.load)
        model_state = torch.load(args.load, map_location="cpu")
        model.load_state_dict(model_state, strict=False)
        model.half().cuda()

    answers = []
    for sample in tqdm(list_data_dict):
        input_text = build_prompt(sample["instruction"], N_SHOT, COT_FLAG)
        generate_kwargs = dict(
            max_new_tokens=128, top_p=0.95, temperature=0.8
        )  # standard use is 256 max_new_tokens
        model_completion = generate(
            model,
            tokenizer,
            input_text,
            generate_kwargs,
            disable_kv_cache=args.disable_kv_cache,
        )
        clean_completaion = model_completion.split("Q:")[0]
        model_answer = clean_answer(model_completion)
        is_cor = is_correct(model_answer, sample["output"])
        answers.append(is_cor)
        if DEBUG:
            print(f"Full input_text:\n{input_text}\n\n")

        print_color = print_green if is_cor else print_red

        print(
            f'Question: {print_color(sample["instruction"])}\n\n'
            f'Answers: {print_color(extract_answer_from_output(sample["output"]))}\n\n'
            f"Model Answers: {print_color(model_answer)}\n\n"
            f"Model Completion: {print_color(clean_completaion)}\n\n"
            f"Is correct: {print_color(is_cor)}\n\n"
        )

        print(
            f"Num of total question: {print_color(len(answers))}, "
            f"Correct num: {print_color(sum(answers))}, "
            f"Accuracy: {print_color(float(sum(answers))/len(answers))}."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "results.txt"), "w") as f:
        for answer in answers:
            print(answer, file=f)

    with open(os.path.join(args.output_dir, "scores.txt"), "w") as f:
        print(
            f"Num of total question: {len(answers)}, "
            f"Correct num: {sum(answers)}, "
            f"Accuracy: {float(sum(answers))/len(answers)}.",
            file=f,
        )


if __name__ == "__main__":
    main()
