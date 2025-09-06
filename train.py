import argparse
import contextlib
import math
import os
import random
import time
from dataclasses import asdict
from statistics import mean

import numpy
import PIL.PngImagePlugin
import torch
import torch.distributed as dist
import torch.optim as optim
from datasets import concatenate_datasets, get_dataset_config_names, load_dataset
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

import models.config as config
import wandb
from data.advanced_datasets import ConstantLengthDataset
from data.collators import VQACollator
from data.data_utils import synchronized_dataloader_step
from data.datasets import VQADataset
from data.processors import get_image_processor, get_tokenizer
from models.gpu_utils import configure_tf32
from models.vision_language_model import VisionLanguageModel

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# Otherwise, the tokenizer will throw a warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Fix for "Decompressed data too large" error with certain PNGs

PIL.PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024


# Set thread limits if configured
def set_thread_limits(max_threads):
    if max_threads is not None and max_threads > 0:
        torch.set_num_threads(max_threads)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(max_threads)
        os.environ["OMP_NUM_THREADS"] = str(max_threads)
        os.environ["MKL_NUM_THREADS"] = str(max_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(max_threads)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(max_threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(max_threads)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def init_dist():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())


def destroy_dist():
    dist.destroy_process_group()


def is_dist():
    return dist.is_available() and dist.is_initialized()


def is_master():
    return dist.get_rank() == 0 if is_dist() else True


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def get_rank():
    return dist.get_rank() if is_dist() else 0


def dist_gather(o):
    o_all = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(o_all, o)
    return o_all


def wrap_model(model):
    return DistributedDataParallel(model, device_ids=[dist.get_rank()])


def get_run_name(train_cfg, vlm_cfg):
    dataset_size = (
        "full_ds"
        if train_cfg.data_cutoff_idx is None
        else f"{train_cfg.data_cutoff_idx}samples"
    )
    batch_size = f"bs{int(train_cfg.batch_size * get_world_size() * train_cfg.gradient_accumulation_steps)}"
    max_training_steps = f"{train_cfg.max_training_steps}"
    learning_rate = f"lr{train_cfg.lr_backbones}-{train_cfg.lr_mp}"
    num_gpus = f"{get_world_size()}xGPU"
    date = time.strftime("%m%d-%H%M%S")
    vit = f"{vlm_cfg.vit_model_type.split('/')[-1]}" + f"_{vlm_cfg.max_img_size}"
    mp = f"mp{vlm_cfg.mp_pixel_shuffle_factor}"
    llm = f"{vlm_cfg.lm_model_type.split('/')[-1]}"

    return f"nanoVLM_{vit}_{mp}_{llm}_{num_gpus}_{dataset_size}_{batch_size}_{max_training_steps}_{learning_rate}_{date}"


def get_dataloaders(train_cfg, vlm_cfg):
    # Create datasets
    # Use single image mode for DINOv3 (resize to 224x224 instead of splitting)
    single_image_mode = vlm_cfg.vit_architecture == "dinov3"
    image_processor = get_image_processor(
        vlm_cfg.max_img_size, vlm_cfg.vit_img_size, single_image_mode
    )
    tokenizer = get_tokenizer(
        vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template
    )

    # Load and combine all training datasets
    combined_train_data = []

    dataset_names_to_load = train_cfg.train_dataset_name
    if "all" in dataset_names_to_load:
        dataset_names_to_load = get_dataset_config_names(train_cfg.train_dataset_path)

    for dataset_name in dataset_names_to_load:
        try:
            train_ds = load_dataset(
                train_cfg.train_dataset_path,
                dataset_name,
                num_proc=min(train_cfg.num_workers, max(1, int(os.cpu_count() // 2)))
                if train_cfg.num_workers > 0
                else 1,
            )
            train_ds["train"][0]  # Check if the dataset is loaded correctly
            combined_train_data.append(train_ds["train"])
        except Exception as e:
            if is_master():
                print(
                    f"Warning: Failed to load dataset config '{dataset_name}' from '{train_cfg.train_dataset_path}'. Error: {e}"
                )
            continue

    if not combined_train_data:
        raise ValueError(
            "No valid datasets were loaded. Please check your dataset path and configurations."
        )

    train_ds = concatenate_datasets(combined_train_data)

    train_ds = train_ds.shuffle(
        seed=0
    )  # Shuffle the training dataset, so train and val get equal contributions from all concatenated datasets

    if is_dist():  # We need to shard the dataset in DDP since we are using an iterable dataset instead of the distributed sampler
        train_ds = train_ds.shard(num_shards=get_world_size(), index=get_rank())

    # Apply cutoff if specified
    if train_cfg.data_cutoff_idx is None:
        total_samples = len(train_ds)  # Use the entire dataset
    else:
        total_samples = min(len(train_ds), train_cfg.data_cutoff_idx)

    val_size = int(total_samples * train_cfg.val_ratio)
    train_size = total_samples - val_size

    train_dataset = VQADataset(
        train_ds.select(range(train_size)),
        tokenizer,
        image_processor,
        vlm_cfg.mp_image_token_length,
    )

    train_dataset = ConstantLengthDataset(
        train_dataset,
        infinite=False,
        max_sample_length=train_cfg.max_sample_length,
        seq_length=vlm_cfg.lm_max_length,
        num_of_sequences=train_cfg.batch_size * 64,
        queue_size=train_cfg.batch_size * 64 * 2,
        max_images_per_example=train_cfg.max_images_per_example,
        max_images_per_knapsack=train_cfg.max_images_per_knapsack,
    )
    val_dataset = VQADataset(
        train_ds.select(range(train_size, total_samples)),
        tokenizer,
        image_processor,
        vlm_cfg.mp_image_token_length,
    )

    # Create collators
    vqa_collator = VQACollator(tokenizer, vlm_cfg.lm_max_length)

    g = torch.Generator()
    g.manual_seed(0)

    # Create dataloaders

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,  # =per device BS in DDP
        collate_fn=vqa_collator,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_sampler = DistributedSampler(
        val_dataset,
        rank=get_rank(),
        num_replicas=get_world_size(),
        shuffle=False,  # Usually False for validation
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        sampler=val_sampler,
        collate_fn=vqa_collator,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return train_loader, val_loader


# Cosine learning rate schedule with warmup (from Karpathy)
# https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py#L353
def get_lr(it, max_lr, max_steps):
    min_lr = max_lr * 0.1
    warmup_steps = max_steps * 0.03
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (
        1.0 + math.cos(math.pi * decay_ratio)
    )  # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


def train(train_cfg, vlm_cfg):
    # Set thread limits before creating dataloaders
    set_thread_limits(train_cfg.max_threads)

    train_loader, val_loader = get_dataloaders(train_cfg, vlm_cfg)

    run_name = get_run_name(train_cfg, vlm_cfg)
    total_dataset_size = len(train_loader.dataset)
    if train_cfg.log_wandb and is_master():
        if train_cfg.data_cutoff_idx is None:
            run_name = run_name.replace("full_ds", f"{total_dataset_size}samples")
    # Initialize model
    if train_cfg.resume_from_vlm_checkpoint:
        model = VisionLanguageModel.from_pretrained(vlm_cfg.vlm_checkpoint_path)
    else:
        model = VisionLanguageModel(
            vlm_cfg, load_backbone=vlm_cfg.vlm_load_backbone_weights
        )

    if train_cfg.log_wandb and is_master():
        run = wandb.init(
            entity=train_cfg.wandb_entity,
            project="nanoVLM",
            config={"VLMConfig": asdict(vlm_cfg), "TrainConfig": asdict(train_cfg)},
            name=run_name,
        )
        # Watch all model gradients and parameters
        wandb.watch(model, log="all", log_freq=train_cfg.wandb_log_steps)

    if is_master():
        print(
            f"nanoVLM initialized with {sum(p.numel() for p in model.parameters()):,} parameters"
        )
        print(
            f"Training summary{' (global)' if is_dist() else ''}: {len(train_loader.dataset)} samples, {int(len(train_loader) * get_world_size())} batches/epoch, batch size {int(train_cfg.batch_size * get_world_size() * train_cfg.gradient_accumulation_steps)}{', training on ' + str(get_world_size()) + ' GPUs' if is_dist() else ''}"
        )
        if is_dist():
            print(
                f"Training summary per GPU: {len(train_loader)} batches/epoch, batch size {train_loader.batch_size}"
            )
        print(
            f"Validation summary{' (global)' if is_dist() else ''}: {len(val_loader.dataset)} samples, {int(len(val_loader) * get_world_size())} batches/epoch, batch size {int(train_cfg.batch_size * get_world_size() * train_cfg.gradient_accumulation_steps)}{', training on ' + str(get_world_size()) + ' GPUs' if is_dist() else ''}"
        )
        if is_dist():
            print(
                f"Validation summary per GPU: {len(val_loader)} batches/epoch, batch size {val_loader.batch_size}"
            )

    # Define optimizer groups
    # Since we have pretrained vision and language backbones, but a newly initialized modality projection layer, it doesn't make sense to train them with the same learning rate
    # You could opt to fully freeze the backbones and only train the MP layer, but finetuning them with a lower learning rate makes the training as a whole easier
    param_groups = [{"params": list(model.MP.parameters()), "lr": train_cfg.lr_mp}]

    # Handle vision encoder freezing/training
    if train_cfg.freeze_vision_encoder:
        # Freeze vision encoder completely (DINOv3 paper's Locked-image Text tuning approach)
        for p in model.vision_encoder.parameters():
            p.requires_grad = False
        if is_master():
            print("Vision encoder weights are frozen (Locked-image Text tuning)")
    elif train_cfg.lr_backbones > 0:
        # Train vision encoder with lower learning rate
        param_groups.append(
            {
                "params": list(model.vision_encoder.parameters()),
                "lr": train_cfg.lr_backbones,
            }
        )
    else:
        # lr_backbones = 0 also freezes vision encoder
        for p in model.vision_encoder.parameters():
            p.requires_grad = False

    # Handle language model training
    if train_cfg.lr_backbones > 0:
        param_groups.append(
            {
                "params": list(model.decoder.parameters()),
                "lr": train_cfg.lr_backbones,
            }
        )
    else:
        for p in model.decoder.parameters():
            p.requires_grad = False

    optimizer = optim.AdamW(param_groups)
    all_params = [p for group in optimizer.param_groups for p in group["params"]]

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    if device.type == "mps":
        torch.backends.mps.enable_fallback_to_cpu = True
        torch.mps.empty_cache()

    print(f"Using device: {device}")
    model.to(device)

    if train_cfg.compile:
        model = torch.compile(model, dynamic=True)
    if is_dist():
        model = wrap_model(model)

    epoch_times = []
    best_val_loss = float("inf")
    global_step = 0
    epoch = 0

    # Training stats accumulators
    accumulated_stats = {
        "tokens_per_second": [],
        "data_load_time": [],
        "fw_bw_time": [],
        "post_process_time": [],
        "images_per_sample": [],
    }

    while global_step < train_cfg.max_training_steps:
        epoch += 1
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0
        total_tokens_processed = 0
        optimizer.zero_grad()
        data_load_start = time.time()

        for i, batch in enumerate(
            synchronized_dataloader_step(train_loader, is_dist())
        ):
            is_update_step = (
                i + 1
            ) % train_cfg.gradient_accumulation_steps == 0 or i + 1 == len(train_loader)
            batch_start_time = time.time()
            images = batch["images"]
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            data_load_time = time.time() - data_load_start

            # When using DDP with gradient accumulation,
            # skip gradient synchronization on intermediate steps to save time.
            # Gradients only need to be synced at the end of each accumulation cycle.
            if (
                is_dist()
                and train_cfg.gradient_accumulation_steps > 1
                and not is_update_step
            ):
                context = model.no_sync()
            else:
                context = contextlib.nullcontext()

            fw_bw_start = time.time()
            autocast_context = torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16
                if device.type in ["cuda", "cpu"]
                else torch.float16,
            )
            with autocast_context:
                with context:
                    _, loss = model(
                        input_ids, images, attention_mask=attention_mask, targets=labels
                    )

            if train_cfg.gradient_accumulation_steps > 1:
                loss = loss / train_cfg.gradient_accumulation_steps

            loss.backward()

            fw_bw_time = time.time() - fw_bw_start
            post_process_start = time.time()
            if is_update_step:
                if train_cfg.max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        all_params, max_norm=train_cfg.max_grad_norm
                    )

                adj_lr_mp = get_lr(
                    global_step, train_cfg.lr_mp, train_cfg.max_training_steps
                )
                optimizer.param_groups[0]["lr"] = adj_lr_mp

                if train_cfg.lr_backbones > 0:
                    adj_lr_backbones = get_lr(
                        global_step,
                        train_cfg.lr_backbones,
                        train_cfg.max_training_steps,
                    )
                    optimizer.param_groups[1]["lr"] = adj_lr_backbones

                optimizer.step()
                optimizer.zero_grad()

            batch_loss = loss.item()
            if train_cfg.gradient_accumulation_steps > 1:
                batch_loss = batch_loss * train_cfg.gradient_accumulation_steps
            total_train_loss += batch_loss

            num_tokens = torch.sum(
                attention_mask
            ).item()  # Sum of attention mask gives number of tokens
            total_tokens_processed += num_tokens
            post_process_time = time.time() - post_process_start

            images_per_sample = [len(image_pack) for image_pack in images]

            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            tokens_per_second = (
                get_world_size() * num_tokens / batch_duration
            )  # Multiply by world size to get global tokens/s

            # Accumulate training stats
            accumulated_stats["tokens_per_second"].append(tokens_per_second)
            accumulated_stats["data_load_time"].append(data_load_time)
            accumulated_stats["fw_bw_time"].append(fw_bw_time)
            accumulated_stats["post_process_time"].append(post_process_time)
            accumulated_stats["images_per_sample"].extend(images_per_sample)

            if (
                train_cfg.eval_in_epochs
                and global_step % train_cfg.eval_interval == 0
                and is_update_step
                and global_step > 0
            ):
                model.eval()
                if device == "cuda":
                    torch.cuda.empty_cache()
                with torch.no_grad():
                    total_val_loss = 0
                    for batch in val_loader:
                        images = batch["images"]
                        input_ids = batch["input_ids"].to(device)
                        labels = batch["labels"].to(device)
                        attention_mask = batch["attention_mask"].to(device)

                        with autocast_context:
                            _, loss = model(
                                input_ids,
                                images,
                                attention_mask=attention_mask,
                                targets=labels,
                            )

                        total_val_loss += loss.item()
                    avg_val_loss = (
                        total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
                    )
                    avg_val_loss = (
                        mean(dist_gather(avg_val_loss)) if is_dist() else avg_val_loss
                    )
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        if is_master():
                            save_model = (
                                model.module if is_dist() else model
                            )  # unwrap the model for saving if DDP
                            save_model.save_pretrained(
                                save_directory=os.path.join(
                                    vlm_cfg.vlm_checkpoint_path, run_name
                                )
                            )

                    lmms_results = {}
                    if train_cfg.use_lmms_eval:
                        from evaluation import cli_evaluate

                        eval_args = argparse.Namespace(
                            model=model.module if is_dist() else model,
                            tasks=train_cfg.lmms_eval_tasks,
                            limit=train_cfg.lmms_eval_limit,
                            batch_size=train_cfg.lmms_eval_batch_size,
                            process_with_media=True,
                            device=device,
                        )
                        # Evaluate using the CLI wrapper
                        eval_results = cli_evaluate(eval_args)

                        if (
                            is_master()
                            and eval_results
                            and "results" in eval_results[0]
                        ):
                            for task_name, task_results in eval_results[0][
                                "results"
                            ].items():
                                for metric_name, metric_value in task_results.items():
                                    if isinstance(metric_value, (int, float)):
                                        lmms_results[
                                            f"{task_name}_{metric_name.split(',')[0]}"
                                        ] = metric_value

                    if is_master():
                        print(
                            f"Step: {global_step}, Val Loss: {avg_val_loss:.4f}, Tokens/s: {tokens_per_second:.2f}"
                        )
                        if train_cfg.log_wandb:
                            run.log(
                                {
                                    "val_loss": avg_val_loss,
                                    **{
                                        f"lmms_eval/{key}": value
                                        for key, value in lmms_results.items()
                                    },
                                },
                                step=global_step,
                            )

                model.train()

            # Log training stats every N steps (ALL RANKS must participate in collective ops)
            if (
                global_step % train_cfg.stats_log_interval == 0
                and len(accumulated_stats["tokens_per_second"]) > 0
                and is_update_step
            ):
                # ALL RANKS: Perform collective operations for training stats
                stats = {}
                for key in [
                    "tokens_per_second",
                    "data_load_time",
                    "fw_bw_time",
                    "post_process_time",
                    "images_per_sample",
                ]:
                    if is_dist():
                        all_values = dist_gather(accumulated_stats[key])
                        all_values_flat = [
                            item for sublist in all_values for item in sublist
                        ]  # Flatten list of lists
                        stats[f"avg_{key}"] = mean(all_values_flat)
                    else:
                        stats[f"avg_{key}"] = mean(accumulated_stats[key])

                for key in [
                    "data_load_time",
                    "fw_bw_time",
                    "post_process_time",
                    "images_per_sample",
                ]:
                    if is_dist():
                        all_values = dist_gather(accumulated_stats[key])
                        all_values_flat = [
                            item for sublist in all_values for item in sublist
                        ]
                        stats[f"max_{key}"] = max(all_values_flat)
                    else:
                        stats[f"max_{key}"] = max(accumulated_stats[key])

                if is_dist():
                    all_images_values = dist_gather(
                        accumulated_stats["images_per_sample"]
                    )
                    all_images_flat = [
                        item for sublist in all_images_values for item in sublist
                    ]
                    stats["min_images_per_sample"] = min(all_images_flat)
                else:
                    stats["min_images_per_sample"] = min(
                        accumulated_stats["images_per_sample"]
                    )

                # MASTER ONLY: Log to wandb
                if (
                    train_cfg.log_wandb
                    and is_master()
                    and global_step % train_cfg.wandb_log_steps == 0
                ):
                    run.log(
                        {
                            **{
                                f"training_stats/{key}": value
                                for key, value in stats.items()
                            },
                        },
                        step=global_step,
                    )

                # ALL RANKS: Reset accumulators
                for key in accumulated_stats:
                    accumulated_stats[key] = []

            # Log batch loss
            if is_update_step:
                # ALL RANKS: gather loss from all ranks if DDP
                if is_dist():
                    batch_loss_gathered = mean(dist_gather(batch_loss))
                else:
                    batch_loss_gathered = batch_loss

                # MASTER ONLY: Log to console and wandb
                if is_master():
                    # Console logging every N steps (configurable)
                    console_log_interval = getattr(
                        train_cfg, "console_log_interval", 100
                    )
                    if global_step % console_log_interval == 0:
                        lr_str = (
                            f"LR: {adj_lr_mp:.2e}" if "adj_lr_mp" in locals() else ""
                        )
                        grad_str = (
                            f"Grad: {grad_norm:.3f}"
                            if train_cfg.max_grad_norm is not None
                            else ""
                        )
                        print(
                            f"Step {global_step}/{train_cfg.max_training_steps} | Loss: {batch_loss_gathered:.4f} | {lr_str} | {grad_str} | Tokens/s: {tokens_per_second:.0f}"
                        )

                    # Wandb logging
                    if (
                        train_cfg.log_wandb
                        and global_step % train_cfg.wandb_log_steps == 0
                    ):
                        run.log(
                            {
                                "batch_loss": batch_loss_gathered,
                                **(
                                    {"grad_norm": grad_norm}
                                    if train_cfg.max_grad_norm is not None
                                    else {}
                                ),
                            },
                            step=global_step,
                        )

            if is_update_step:
                global_step += 1

                # Save checkpoint at regular intervals if configured
                if (
                    train_cfg.save_checkpoint_steps is not None
                    and global_step % train_cfg.save_checkpoint_steps == 0
                    and is_master()
                ):
                    save_model = model.module if is_dist() else model
                    checkpoint_dir = os.path.join(
                        vlm_cfg.vlm_checkpoint_path, run_name, f"step_{global_step}"
                    )
                    save_model.save_pretrained(save_directory=checkpoint_dir)
                    print(f"Saved checkpoint at step {global_step} to {checkpoint_dir}")

                if global_step >= train_cfg.max_training_steps:
                    break
            data_load_start = time.time()

        avg_train_loss = total_train_loss / len(train_loader)
        # gather average batch loss from all ranks if DDP
        avg_train_loss = (
            mean(dist_gather(avg_train_loss)) if is_dist() else avg_train_loss
        )

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        # gather and sum total_tokens_processed across all ranks if DDP
        total_tokens_processed = (
            sum(dist_gather(total_tokens_processed))
            if is_dist()
            else total_tokens_processed
        )
        epoch_tokens_per_second = total_tokens_processed / epoch_duration

        if is_master():
            if train_cfg.log_wandb:
                run.log(
                    {
                        "epoch_loss": avg_train_loss,
                        "epoch_duration": epoch_duration,
                        "epoch_tokens_per_second": epoch_tokens_per_second,
                    }
                )

            print(
                f"Epoch: {epoch}, Step: {global_step}/{train_cfg.max_training_steps}, Train Loss: {avg_train_loss:.4f} | Time: {epoch_duration:.2f}s | T/s: {epoch_tokens_per_second:.2f}"
            )

    # Summary Statistics
    if is_master():
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        total_training_time = sum(epoch_times)
        batch_size = int(
            train_cfg.batch_size
            * get_world_size()
            * train_cfg.gradient_accumulation_steps
        )
        total_samples_processed = batch_size * global_step
        avg_time_per_sample = total_training_time / total_samples_processed
        print(f"Average time per epoch: {avg_epoch_time:.2f}s")
        print(f"Average time per sample: {avg_time_per_sample:.4f}s")

        # Push the best model to the hub (Please set your user name in the config!)
        if vlm_cfg.hf_repo_name is not None:
            print("Training complete. Pushing model to Hugging Face Hub...")
            hf_model = VisionLanguageModel.from_pretrained(
                os.path.join(vlm_cfg.vlm_checkpoint_path, run_name)
            )
            hf_model.push_to_hub(vlm_cfg.hf_repo_name)

        if train_cfg.log_wandb:
            run.summary["avg_epoch_time"] = avg_epoch_time
            run.summary["avg_time_per_sample"] = avg_time_per_sample
            run.finish()


def main():
    # Load default configs to show in help
    default_train_cfg = config.TrainConfig()
    default_vlm_cfg = config.VLMConfig()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="If no arguments are passed, uses original 460M siglip+smolLM2 configuration",
    )

    # Model selection arguments
    parser.add_argument(
        "--vision_encoder",
        type=str,
        choices=["siglip", "dinov3", "dinov2"],
        default="siglip",
        help="Vision encoder architecture to use",
    )
    parser.add_argument(
        "--dinov3_model",
        type=str,
        choices=["vits16plus", "vitb16", "vitl14"],
        default="vits16plus",
        help="DINOv3 model size to use (only applies when --vision_encoder=dinov3)",
    )
    parser.add_argument(
        "--language_model",
        type=str,
        choices=["llama", "smollm", "gemma"],
        default="smollm",
        help="Language model architecture to use",
    )
    parser.add_argument(
        "--use_preset",
        type=str,
        choices=["dinov3_gemma", None],
        default=None,
        help="Use a preset configuration (overrides vision_encoder and language_model)",
    )

    # Training arguments
    parser.add_argument(
        "--lr_mp",
        type=float,
        default=default_train_cfg.lr_mp,
        help="Learning rate for the mapping network",
    )
    parser.add_argument(
        "--lr_backbones",
        type=float,
        default=default_train_cfg.lr_backbones,
        help="Learning rate for the backbones",
    )
    parser.add_argument(
        "--vlm_checkpoint_path",
        type=str,
        default=default_vlm_cfg.vlm_checkpoint_path,
        help="Path to the VLM checkpoint for loading or saving",
    )
    parser.add_argument(
        "--compile", action="store_true", help="Use torch.compile to optimize the model"
    )
    parser.add_argument(
        "--log_wandb",
        action="store_true",
        help="Force enable wandb logging (use --no_log_wandb to disable)",
    )
    parser.add_argument(
        "--resume_from_vlm_checkpoint",
        action="store_true",
        help="Resume training from VLM checkpoint specified by vlm_checkpoint_path",
    )
    parser.add_argument(
        "--freeze_vision_encoder",
        action="store_true",
        help="Freeze vision encoder weights (Locked-image Text tuning as per DINOv3 paper)",
    )
    parser.add_argument(
        "--no_log_wandb", action="store_true", help="Do not log to wandb"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=default_train_cfg.num_workers,
        help="Number of DataLoader worker threads, 0 disables multiprocessing",
    )
    parser.add_argument(
        "--max_threads",
        type=int,
        default=default_train_cfg.max_threads,
        help="Maximum number of threads for torch operations",
    )
    parser.add_argument(
        "--console_log_interval",
        type=int,
        default=default_train_cfg.console_log_interval,
        help="Print loss to console every N steps",
    )
    parser.add_argument(
        "--save_checkpoint_steps",
        type=int,
        default=default_train_cfg.save_checkpoint_steps,
        help="Save checkpoint every N steps regardless of validation loss",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=default_train_cfg.eval_interval,
        help="Evaluate and potentially save best model every N steps",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=default_train_cfg.batch_size,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=default_train_cfg.gradient_accumulation_steps,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max_training_steps",
        type=int,
        default=default_train_cfg.max_training_steps,
        help="Maximum number of training steps",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=default_train_cfg.max_grad_norm,
        help="Maximum gradient norm for clipping",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=default_train_cfg.wandb_entity,
        help="Wandb entity name (leave empty for your default entity)",
    )

    args = parser.parse_args()

    # Select configuration based on arguments
    if args.use_preset == "dinov3_gemma":
        vlm_cfg = config.get_dinov3_gemma_config()
    else:
        vlm_cfg = config.VLMConfig()

        # Update vision encoder configuration
        if args.vision_encoder in ["dinov3", "dinov2"]:
            vlm_cfg.vit_architecture = "dinov3"
            if args.vision_encoder == "dinov2":
                vlm_cfg.vit_model_type = "facebook/dinov2-small"
                vlm_cfg.vit_num_registers = 0  # DINOv2 doesn't have registers
                vlm_cfg.vit_use_swiglu = False  # DINOv2 doesn't use SwiGLU
                vlm_cfg.vit_use_rope = False
            else:
                # Real DINOv3!
                # Support different DINOv3 model sizes
                dinov3_models = {
                    "vits16plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
                    "vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
                    "vitl14": "facebook/dinov3-vitl14-pretrain-lvd1689m",
                }
                vlm_cfg.vit_model_type = dinov3_models.get(
                    args.dinov3_model, "facebook/dinov3-vits16plus-pretrain-lvd1689m"
                )
                vlm_cfg.vit_num_registers = 4  # DINOv3 has registers
                # Only "plus" models use SwiGLU
                vlm_cfg.vit_use_swiglu = "plus" in args.dinov3_model
                vlm_cfg.vit_use_rope = True  # DINOv3 uses RoPE
                vlm_cfg.vit_img_size = 224  # DINOv3 default size
            vlm_cfg.vit_cls_flag = True
            vlm_cfg.mp_handle_special_tokens = True
            # For 224x224 images with patch_size=16, we get 14x14 patches
            # Need pixel_shuffle_factor=2 (not 4) since 14 % 4 != 0
            vlm_cfg.mp_pixel_shuffle_factor = 2
            vlm_cfg.mp_image_token_length = 49  # (14/2)^2 = 7^2 = 49

        # Update language model configuration
        if args.language_model == "gemma":
            vlm_cfg.lm_architecture = "gemma"
            vlm_cfg.lm_model_type = "google/gemma-3-270m-it"  # Real Gemma 3 270M model
            vlm_cfg.lm_tokenizer = "google/gemma-3-270m-it"
            vlm_cfg.lm_base_vocab_size = 262144  # Gemma-3 actual vocab size
            # Recalculate vocab size with extra tokens
            vlm_cfg.lm_vocab_size = config.round_up_to_multiple(
                vlm_cfg.lm_base_vocab_size + vlm_cfg.extra_token_amount
            )

    train_cfg = config.TrainConfig()

    # Apply training configuration arguments
    train_cfg.lr_mp = args.lr_mp
    train_cfg.lr_backbones = args.lr_backbones
    train_cfg.batch_size = args.batch_size
    train_cfg.gradient_accumulation_steps = args.gradient_accumulation_steps
    train_cfg.max_training_steps = args.max_training_steps
    train_cfg.max_grad_norm = args.max_grad_norm
    train_cfg.num_workers = args.num_workers
    train_cfg.max_threads = args.max_threads
    train_cfg.console_log_interval = args.console_log_interval
    train_cfg.eval_interval = args.eval_interval
    train_cfg.save_checkpoint_steps = args.save_checkpoint_steps
    train_cfg.wandb_entity = args.wandb_entity

    # Apply VLM configuration arguments
    vlm_cfg.vlm_checkpoint_path = args.vlm_checkpoint_path

    # Handle boolean flags
    train_cfg.compile = args.compile
    if args.no_log_wandb:
        train_cfg.log_wandb = False
    elif args.log_wandb:
        train_cfg.log_wandb = True

    train_cfg.freeze_vision_encoder = args.freeze_vision_encoder
    if args.freeze_vision_encoder and args.use_preset == "dinov3_gemma":
        if is_master():
            print("Using frozen vision encoder with DINOv3 (recommended by paper)")

    train_cfg.resume_from_vlm_checkpoint = args.resume_from_vlm_checkpoint
    if args.resume_from_vlm_checkpoint:
        # When resuming a full VLM, we don't need to load individual backbone weights from original sources
        vlm_cfg.vlm_load_backbone_weights = False

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        init_dist()

    # Configure TF32 for better performance on Ampere+ GPUs
    configure_tf32()

    if is_master():
        print("--- VLM Config ---")
        print(vlm_cfg)
        print("--- Train Config ---")
        print(train_cfg)

    train(train_cfg, vlm_cfg)

    if is_dist():
        destroy_dist()


if __name__ == "__main__":
    main()
