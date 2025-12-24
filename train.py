import math
import os
import copy
import torch
from time import time
from json import dumps
from typing import Optional
from dataclasses import asdict
import torch.nn.functional as F
import torch.distributed as dist
from argparse import ArgumentParser

import wandb
from lm_eval.utils import make_table
from safetensors.torch import save_file
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from eval import evaluate
from configs import Config
from optim import get_scheduler
from model import OnlineZZModel
from generate import z2z_generate, GenerateConfig
from data import DataLoaderLite, tokenize_and_compress_dataset
from utils import (
    nanoid,
    dataclass_from_file,
    setup_distributed,
    setup_seed,
    setup_wandb,
    wandb_log,
    print_trainable_parameters,
    flatten,
    to_str_dict,
    get_seed,
    upload_checkpoints,
    find_latest_checkpoint,
)


def validate(
    model: OnlineZZModel,
    val_loader: DataLoaderLite,
    config: Config,
    master_process: bool,
    ddp: bool,
) -> torch.Tensor:
    tokenizer = model.tokenizer
    model.eval()
    val_loader.reset()
    with torch.no_grad():
        total_val_loss = torch.tensor(0.0, device=device)
        total_val_CLM_loss = torch.tensor(0.0, device=device)
        total_base_token_val_CLM_loss = torch.tensor(0.0, device=device)
        total_hyper_token_val_CLM_loss = torch.tensor(0.0, device=device)
        total_val_AE_loss = torch.tensor(0.0, device=device)
        for _ in range(config.val_steps):
            input_ids, labels, codebook, _ = val_loader.next_batch()

            non_padding_token_mask = labels != int(
                tokenizer.pad_token_id
            )  # shape (B, S)

            if config.disable_CLM:
                step_per_token_CLM_loss = torch.tensor(0.0, device=device)
                step_per_base_token_CLM_loss = torch.tensor(0.0, device=device)
                step_per_hyper_token_CLM_loss = torch.tensor(0.0, device=device)
            else:
                logits, metadata = model.forward(input_ids, codebook)  # (B, T, V+V_E)
                CLM_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),  # (B * T, V+V_E)
                    labels.reshape(-1),  # (B * T,)
                    ignore_index=tokenizer.pad_token_id,  # type: ignore
                    reduction="none",
                )  # (B * T)
                step_per_token_CLM_loss = CLM_loss.sum() / non_padding_token_mask.sum()

                base_token_mask = (
                    labels.reshape(-1) < config.initial_vocab_size
                ) * non_padding_token_mask.reshape(-1)
                hyper_token_mask = (
                    ~base_token_mask.bool()
                ) * non_padding_token_mask.reshape(-1)

                step_per_base_token_CLM_loss = (
                    CLM_loss * base_token_mask
                ).sum() / base_token_mask.sum()
                step_per_hyper_token_CLM_loss = (
                    CLM_loss * hyper_token_mask
                ).sum() / hyper_token_mask.sum()

            if config.embedding_encoder.auto_encoder_loss_alpha > 0.0:
                B, V_E, M = codebook.size()
                if config.disable_CLM:
                    _, metadata = model.hyper_embedding.forward(input_ids, codebook)
                else:
                    metadata = metadata["embedding_metadata"]
                AE_logits = metadata["AE_logits"]  # shape (B*V_E, M+1, V)
                # mask = codebook != tokenizer.pad_token_id  # shape (B, V_E, M)
                non_padding_token_mask = torch.ne(
                    codebook, tokenizer.pad_token_id
                )  # shape (B, V_E, M)
                CE_labels = codebook.reshape(-1)  # shape (B*V_E*M)
                # Use Cross Entropy Loss
                AE_loss = F.cross_entropy(
                    AE_logits[:, :-1, :].reshape(-1, AE_logits.size(-1)),
                    CE_labels,
                    ignore_index=tokenizer.pad_token_id,  # type: ignore
                    reduction="none",
                )
                # shape (B*V_E*M)
                AE_loss *= non_padding_token_mask.reshape(-1)  # shape (B*V_E*M)
                step_token_AE_loss = AE_loss.sum() / non_padding_token_mask.sum()
            else:
                step_token_AE_loss = torch.tensor(0.0, device=device)
            step_loss = (
                step_token_AE_loss * config.embedding_encoder.auto_encoder_loss_alpha
                + step_per_token_CLM_loss
                * (1 - config.embedding_encoder.auto_encoder_loss_alpha)
            )

            total_val_loss += step_loss.detach()
            total_val_CLM_loss += step_per_token_CLM_loss.detach()
            total_base_token_val_CLM_loss += step_per_base_token_CLM_loss.detach()
            total_hyper_token_val_CLM_loss += step_per_hyper_token_CLM_loss.detach()
            total_val_AE_loss += step_token_AE_loss.detach()

        total_val_loss /= config.val_steps
        total_val_CLM_loss /= config.val_steps
        total_base_token_val_CLM_loss /= config.val_steps
        total_hyper_token_val_CLM_loss /= config.val_steps
        total_val_AE_loss /= config.val_steps
        if ddp:
            dist.all_reduce(total_val_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(total_val_CLM_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(total_base_token_val_CLM_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(total_hyper_token_val_CLM_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(total_val_AE_loss, op=dist.ReduceOp.AVG)

        if master_process:
            print(f"validation loss: {total_val_loss.item():.4f}")

            wandb_log(
                config,
                {
                    "val_total_loss": total_val_loss.item(),
                    "val_CLM_loss": total_val_CLM_loss.item(),
                    "base_token_val_ppl": torch.exp(
                        total_base_token_val_CLM_loss
                    ).item(),
                    "hyper_token_val_ppl": torch.exp(
                        total_hyper_token_val_CLM_loss
                    ).item(),
                    "val_AE_loss": total_val_AE_loss.item(),
                },
            )

    return total_val_loss


def checkpoint(
    model: OnlineZZModel,
    config: Config,
    step: int,
    run_id: str,
    optimizer: torch.optim.AdamW,
    scheduler: any,
) -> None:
    checkpoint_dir = os.path.join(config.checkpoint_dir, run_id)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    metadata = {
        "seed": f"{get_seed()}",
        "step": f"{step:05d}",
        **to_str_dict(flatten(asdict(config)), "config"),
    }

    if val_loss is not None:
        metadata["val_loss"] = f"{val_loss.item():.4f}"

    # Save model parameters
    save_file(
        {
            name: param
            for name, param in model.named_parameters()
            if param.requires_grad
            # since 04/10, we save all the parameters to support resume from prev trains
            # and "vXtU" not in name  # vXtU is here to avoid saving the decoder
        },
        os.path.join(checkpoint_dir, f"model_{step}.safetensors"),
        metadata=metadata,
    )

    # Save training state
    torch.save(
        {
            "step": step,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict()
            if hasattr(scheduler, "state_dict")
            else None,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state()
            if torch.cuda.is_available()
            else None,
        },
        os.path.join(checkpoint_dir, f"training_state_{step}.pt"),
    )


def load_checkpoint(
    model: OnlineZZModel,
    config: Config,
    run_id: str,
    step: int,
    optimizer: torch.optim.AdamW,
    scheduler: any,
    device: torch.device,
) -> None:
    """
    Load checkpoint and return current step and training history
    """
    checkpoint_dir = os.path.join(config.checkpoint_dir, run_id)

    # Load model parameters
    ckp_config = model._load_checkpoint(
        os.path.join(checkpoint_dir, f"model_{step}.safetensors")
    )
    if ckp_config != config:
        raise ValueError("Checkpoint config does not match current config.")

    # Load training state
    training_state_path = os.path.join(checkpoint_dir, f"training_state_{step}.pt")
    state = torch.load(training_state_path, map_location=device)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    if state["scheduler_state_dict"] and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(state["scheduler_state_dict"])

    # Restore RNG state
    torch.set_rng_state(state["rng_state"].cpu())
    if torch.cuda.is_available() and state["cuda_rng_state"] is not None:
        torch.cuda.set_rng_state(state["cuda_rng_state"].cpu(), device=device)

    return None


if __name__ == "__main__":
    (
        ddp,
        ddp_rank,
        ddp_local_rank,
        ddp_world_size,
        master_process,
        device,
        device_type,
    ) = setup_distributed()

    torch.set_float32_matmul_precision("high")

    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, required=False, default=42)
    parser.add_argument(
        "--strict_deterministic", type=bool, required=False, default=False
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--tasks",
        type=str,
        required=False,
        default="wikitext,pile_10k,paloma_mc4",
    )
    parser.add_argument("--limit", type=int, required=False, default=20)
    parser.add_argument(
        "--rid", type=str, required=False, help="Specific run ID to use/resume"
    )
    args = parser.parse_args()
    # Get run_id before setup_seed, otherwise it becomes deterministic
    # Either use provided run_id or create new one
    run_id = args.rid if args.rid else nanoid()
    if master_process:
        print(f"Using run_id: {run_id}")

    setup_seed(args.seed, args.strict_deterministic)
    config = dataclass_from_file(Config, args.config)

    # Find latest checkpoint for this run
    latest_step = find_latest_checkpoint(config.checkpoint_dir, run_id)
    if latest_step is not None and master_process:
        print(f"Found checkpoint at step {latest_step}")

    lm_eval_task_table = wandb.Table(columns=["ckpt"] + args.tasks.split(","))
    text_generation_table = wandb.Table(columns=["ckpt", "response"])
    resume_run_table = wandb.Table(columns=["resumed_from_step"])

    if master_process:
        print(f"run_id: {run_id}")
        # Initialize wandb with consistent run_id in case of resume
        if config.wandb_config is not None:
            setup_wandb(config, run_id)

    gradient_accumulation_steps = math.ceil(
        config.total_batch_size
        / (config.per_device_batch_size * config.seq_length * ddp_world_size)
    )

    try:
        print("Trying to load dataset...")
        train_loader = DataLoaderLite(
            config, "train", device, ddp_local_rank, ddp_world_size
        )

    except FileNotFoundError:
        raise RuntimeError(
            "Dataset not found. Please run `python data.py --config xxx` to generate the dataset."
        )

    # Load validation dataset if needed
    val_loader = (
        DataLoaderLite(config, "validation", device, ddp_local_rank, ddp_world_size)
        if config.val_steps > 0
        else None
    )

    if config.max_steps is None:
        config.max_steps = (
            config.epochs
            * ddp_world_size
            * len(train_loader.indices)
            * config.seq_length
            // config.total_batch_size
        )

    if master_process:
        total_train_tokens = (
            ddp_world_size * len(train_loader.indices) * config.seq_length
        )
        required_train_tokens = config.total_batch_size * config.max_steps
        num_epochs = required_train_tokens / total_train_tokens
        print("-" * 45)
        print(f"Number of available tokens: {total_train_tokens:>15,}")
        print(f"Number of required tokens: {required_train_tokens:>16,}")
        print(f"Number of epochs: {num_epochs:>25.2f}")
        print("-" * 45)
    if config.pretrained_adapter_path is not None:
        model = OnlineZZModel.load_pretrained(
            config.pretrained_adapter_path,
            config.pretrained_hub_adapter_path,
            device=device,
            extra_vocab_size=config.extra_vocab_size,
        )  # .to(config.dtype)
        assert (
            model.config.pretrained_tokenizer_name_or_path
            == config.pretrained_tokenizer_name_or_path
        ), f"Tokenizer mismatch: {model.config.pretrained_tokenizer_name_or_path} != {config.pretrained_tokenizer_name_or_path}"

        assert (
            model.config.lora == config.lora
        ), f"LoRA config mismatch: {model.config.lora} != {config.lora}"
        # TODO probably we should check embedding_encoder config as well
        # assert model.config.embedding_encoder == config.embedding_encoder, f"Embedding encoder config mismatch: {model.config.embedding_encoder} != {config.embedding_encoder}"

    else:
        model = OnlineZZModel(config, device)  # .to(config.dtype)

    tokenizer = model.tokenizer

    scheduler = get_scheduler(config)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=scheduler(0),
        fused=device_type == "cuda",
    )

    if master_process:
        print_trainable_parameters(model)

    if master_process:
        # For early stopping from training
        train_loss_history: list[float] = []
        min_improvement = 0.01
        early_stopping_patience = config.early_stopping_patience
        checkpoint_train_losses: list[float] = []
        sofar_best_train_loss = float("inf")

    stop_training = torch.tensor(0, device="cuda")  # 0 means continue training

    # Try to load latest checkpoint if it exists
    start_step = 0
    if latest_step is not None:
        if master_process:
            print(f"Resuming from checkpoint at step {latest_step}")
        load_checkpoint(
            model, config, run_id, latest_step, optimizer, scheduler, device
        )
        start_step = latest_step  # Start from latest step

    if master_process:
        resume_run_table.add_data(latest_step)
        new_resume_run_table = wandb.Table(
            columns=resume_run_table.columns, data=resume_run_table.data
        )
        wandb_log(config=config, items={"resume_run": new_resume_run_table})

    if config.float8_base_model:
        model.convert_to_float8_training()

    if config.compile_model and device_type == "cuda":
        model.torch_compile()

    if ddp:
        model = DDP(
            model, device_ids=[ddp_local_rank], find_unused_parameters=True
        )  # find_unused_parameters=True Is this needed?
    raw_model = model.module if ddp else model

    # Modify training loop to start from the correct step
    max_steps = config.max_steps

    step = 0
    if step < start_step:
        # fast forward to the start step by iterating over the train and val loaders
        for update_step in range(start_step - step):
            for micro_step in range(gradient_accumulation_steps):
                train_loader.next_batch()
            if (
                val_loader
                and (config.val_steps > 0 and config.val_interval > 0)
                and update_step > 0
                and update_step % config.val_interval == 0
            ):
                for _ in range(config.val_steps):
                    val_loader.next_batch()

    for step in range(start_step, max_steps):

        if (
            val_loader
            and (config.val_steps > 0 and config.val_interval > 0)
            and (step > start_step and (step % config.val_interval == 0))
        ):
            val_loss = validate(
                raw_model,
                val_loader,
                config,
                master_process,
                ddp,
            )

            if config.generation_prompt is not None and master_process:
                generate_config = GenerateConfig(
                    max_new_tokens=100,
                    ddp_rank=ddp_rank,
                    top_k=0 if ddp_rank == 0 else 50,
                )

                full_text, _, _, _, _ = z2z_generate(
                    config.generation_prompt, raw_model, generate_config
                )
                text_generation_table.add_data(
                    step,
                    full_text,
                )
                new_text_generation_table = wandb.Table(
                    columns=text_generation_table.columns,
                    data=text_generation_table.data,
                )
                wandb_log(
                    config=config,
                    items={"text_generation": new_text_generation_table},
                )

                print(
                    f"rank 0 (deterministic) | response: {dumps(full_text, ensure_ascii=False)}"
                )

        else:
            val_loss = None

        if (
            master_process
            and step > start_step
            and (step % config.checkpoint_interval == 0)
        ):

            checkpoint(raw_model, config, step, run_id, optimizer, scheduler)
            results = evaluate(raw_model, args.tasks, args.limit)
            _tasks = args.tasks.split(",")
            scores = [
                results["results"][task]["bits_per_byte,none"]
                for task in _tasks
                if task in results["results"]
            ]

            lm_eval_task_table.add_data(*([step] + scores))
            new_lm_eval_task_table = wandb.Table(
                columns=lm_eval_task_table.columns, data=lm_eval_task_table.data
            )
            # why a copy? see github.com/wandb/wandb/issues/2981
            wandb_log(config=config, items={"tasks": new_lm_eval_task_table})

            # Put early stopping here
            curr_train_loss = torch.mean(
                torch.tensor(train_loss_history[-config.checkpoint_interval :])
            ).item()
            checkpoint_train_losses.append(curr_train_loss)
            if curr_train_loss < sofar_best_train_loss - min_improvement:
                sofar_best_train_loss = curr_train_loss
                early_stopping_patience = config.early_stopping_patience
                print(
                    f"New best train loss: {sofar_best_train_loss:.6f}, patience reset to {early_stopping_patience}"
                )

            else:
                if early_stopping_patience == 0:
                    print(f"Early stopping, best train loss: {sofar_best_train_loss}")
                    stop_training = torch.tensor(
                        1, device="cuda"
                    )  # 1 means stop training
                else:
                    early_stopping_patience -= 1
                    print(
                        f"Not enough improvement, patience left: {early_stopping_patience+1} -> {early_stopping_patience}"
                    )

        # Synchronize the stopping decision across all processes
        if dist.is_initialized():
            dist.all_reduce(stop_training, op=dist.ReduceOp.MAX)

        # If any process decides to stop, all must stop
        if stop_training.item() > 0:
            print("Early stopping triggered across all processes.")
            break

        t0 = time()
        model.train()
        batch_hypertokens_ratio = torch.tensor(0.0, device=device)
        batch_padding_efficiency = torch.tensor(0.0, device=device)
        for micro_step in range(gradient_accumulation_steps):
            input_ids, labels, codebook, _ = train_loader.next_batch()

            non_padding_token_mask = input_ids != int(
                tokenizer.pad_token_id
            )  # shape (B, S)
            batch_padding_efficiency += non_padding_token_mask.sum() / input_ids.numel()

            hypertokens_ratio = (
                (input_ids > config.initial_vocab_size).sum()
                / non_padding_token_mask.sum()
                if non_padding_token_mask.sum() > 0
                else torch.tensor(0.0, device=device)
            )
            batch_hypertokens_ratio += hypertokens_ratio

            if ddp:
                model.require_backward_grad_sync = (
                    micro_step == gradient_accumulation_steps - 1
                )

            if config.disable_CLM:
                per_token_CLM_loss = torch.tensor(0.0, device=device)
            else:
                logits, metadata = model(input_ids, codebook)

                CLM_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=tokenizer.pad_token_id,  # type: ignore
                    reduction="none",
                )
                per_token_CLM_loss = CLM_loss.sum() / non_padding_token_mask.sum()

            if config.embedding_encoder.auto_encoder_loss_alpha > 0.0:
                B, V_E, M = codebook.size()
                if config.disable_CLM:
                    _, metadata = raw_model.hyper_embedding.forward(input_ids, codebook)
                else:
                    metadata = metadata["embedding_metadata"]
                AE_logits = metadata["AE_logits"]  # shape (B*V_E, M+1, V)
                non_padding_token_mask = codebook != int(tokenizer.pad_token_id)  # type: ignore # shape (B, V_E, M)

                CE_labels = codebook.reshape(-1)  # shape (B*V_E*M)
                # Use Cross Entropy Loss
                AE_loss = F.cross_entropy(
                    AE_logits[:, :-1, :].reshape(-1, AE_logits.size(-1)),
                    CE_labels,
                    ignore_index=tokenizer.pad_token_id,  # type: ignore
                    reduction="none",
                )
                # shape (B*V_E*M)
                AE_loss *= non_padding_token_mask.reshape(-1)  # shape (B*V_E*M)
                per_token_AE_loss = AE_loss.sum() / non_padding_token_mask.sum()
            else:
                per_token_AE_loss = torch.tensor(0.0, device=device)
            loss = (
                per_token_AE_loss * config.embedding_encoder.auto_encoder_loss_alpha
                + per_token_CLM_loss
                * (1 - config.embedding_encoder.auto_encoder_loss_alpha)
            )

            loss.backward()

        train_loss = loss.detach()
        train_CLM_loss = per_token_CLM_loss.detach()
        train_AE_loss = per_token_AE_loss.detach()
        for p in model.parameters():
            if p.grad is not None:
                p.grad /= gradient_accumulation_steps

        hypertokens_ratio = batch_hypertokens_ratio / gradient_accumulation_steps
        padding_efficiency = batch_padding_efficiency / gradient_accumulation_steps

        if ddp:
            dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(train_CLM_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(train_AE_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(hypertokens_ratio, op=dist.ReduceOp.AVG)
            dist.all_reduce(padding_efficiency, op=dist.ReduceOp.AVG)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = scheduler(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()

        model.zero_grad(set_to_none=True)

        if device_type == "cuda":
            torch.cuda.synchronize()

        dt = time() - t0
        tokens_processed = (
            config.per_device_batch_size
            * config.seq_length
            * gradient_accumulation_steps
            * ddp_world_size
        )
        tokens_per_sec = tokens_processed / dt
        if master_process:
            print(
                f"step {step:5d} | loss: {train_loss.item():.6f} | CLM_loss: {train_CLM_loss.item():.6f} | AE_loss: {train_AE_loss.item():.6f} | lr {lr:.4e} | norm: {gradient_norm:.4f} | dt: {dt*1000:.2f}ms | tok_per_sec: {tokens_per_sec:.2f} | hypertokens_ratio: {hypertokens_ratio.item():.2%} | padding_efficiency: {padding_efficiency.item():.2%}"
            )

            wandb_log(
                config,
                {
                    "loss": train_loss.item(),
                    "CLM_loss": train_CLM_loss.item(),
                    "AE_loss": train_AE_loss.item(),
                    "lr": lr,
                    "norm": gradient_norm,
                    "hypertokens_ratio": hypertokens_ratio * 100,
                    "padding_efficiency": padding_efficiency * 100,
                },
            )
            train_loss_history.append(train_loss.item())

    if master_process:
        print(
            f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB"
        )

    if ddp:
        destroy_process_group()

    if master_process:
        upload_checkpoints(config, run_id, only_last_checkpoint=True)

    # evaluate the model
    if master_process and args.tasks is not None:
        results = evaluate(raw_model, args.tasks, args.limit)

        if results is not None:
            print(make_table(results))
