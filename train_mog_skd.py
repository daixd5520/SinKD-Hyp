#!/usr/bin/env python
"""
Training script with MoG-SKD framework

This script demonstrates how to use MoG-SKD for knowledge distillation.
"""

import argparse
import logging
import os
import random
import csv
import math
from dataclasses import dataclass
from typing import Optional, Union
from itertools import chain

import datasets
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    default_data_collator,
    DataCollatorForSeq2Seq,
    AdamW,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import PaddingStrategy
from promptsource.templates import DatasetTemplates

# Import MoG-SKD framework
from mog_skd import MoGSKD, MoGSKDConfig


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning T0 with MoG-SKD")

    # Dataset and template
    parser.add_argument("-d", "--dataset_name", type=str, required=True,
                       help="Dataset name")
    parser.add_argument("-s", "--dataset_config_name", type=str, default=None,
                       help="Dataset config name")
    parser.add_argument("-t", "--template_name", type=str, required=True,
                       help="Template name from promptsource")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                       help="Output directory")

    # Model
    parser.add_argument("-m", "--model_name_or_path", type=str, required=True,
                       help="Student model path or name")
    parser.add_argument("--teacher_model_path", type=str, default="",
                       help="Teacher model path")

    # MoG-SKD specific
    parser.add_argument("--use_mog_skd", action="store_true",
                       help="Use MoG-SKD instead of standard distillation")
    parser.add_argument("--lambda_reg", type=float, default=0.1,
                       help="Entropy regularization coefficient for gating")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Distillation temperature")
    parser.add_argument("--use_sinkhorn", action="store_true",
                       help="Use Sinkhorn in Euclidean expert")
    parser.add_argument("--learnable_curvature", action="store_true",
                       help="Make hyperbolic curvature learnable")
    parser.add_argument("--hyperbolic_c", type=float, default=1.0,
                       help="Initial hyperbolic curvature")

    # Training
    parser.add_argument("-tb", "--per_device_train_batch_size", type=int, default=4,
                       help="Train batch size")
    parser.add_argument("-eb", "--per_device_eval_batch_size", type=int, default=8,
                       help="Eval batch size")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("-ep", "--num_train_epochs", type=int, default=10,
                       help="Number of epochs")
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("-ga", "--gradient_accumulation_steps", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("-ws", "--num_warmup_steps", type=int, default=0,
                       help="Warmup steps")
    parser.add_argument("-sd", "--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("-ns", "--num_shots", type=int, default=None,
                       help="Number of shots for few-shot learning")

    # Other
    parser.add_argument("-il", "--max_length", type=int, default=1024,
                       help="Max input length")
    parser.add_argument("-tl", "--target_max_length", type=int, default=256,
                       help="Max target length")
    parser.add_argument("-db", "--debug", action="store_true",
                       help="Debug mode (use subset)")
    parser.add_argument("-wb", "--wandb_proj", type=str, default=None,
                       help="W&B project name")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    # Initialize accelerator
    accelerator = Accelerator()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)

    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Load dataset
    if args.dataset_name == "anli":
        raw_train_dataset = load_dataset(args.dataset_name, split=f'train_{args.dataset_config_name}')
        raw_eval_dataset = load_dataset(args.dataset_name, split=f'dev_{args.dataset_config_name}')
    else:
        raw_train_dataset = load_dataset("", "copa", split="train",
                                        cache_dir="", download_mode="reuse_cache_if_exists")
        raw_eval_dataset = load_dataset("", "copa", split="validation",
                                       cache_dir="", download_mode="reuse_cache_if_exists")

    # Debug mode
    if args.debug:
        raw_train_dataset = raw_train_dataset.select(range(min(32, len(raw_train_dataset))))
        raw_eval_dataset = raw_eval_dataset.select(range(min(32, len(raw_eval_dataset))))

    column_names = raw_eval_dataset.column_names

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Load student model
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError("Need config or model path")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    # Load teacher model
    teacher_model = AutoModelForSeq2SeqLM.from_pretrained(args.teacher_model_path)
    teacher_model.eval()  # Teacher in eval mode

    # Initialize MoG-SKD if enabled
    if args.use_mog_skd:
        mog_skd_config = MoGSKDConfig(
            T=args.temperature,
            lambda_reg=args.lambda_reg,
            hidden_dim=32,
            use_sinkhorn=args.use_sinkhorn,
            learnable_curvature=args.learnable_curvature,
            hyperbolic_c=args.hyperbolic_c
        )
        mog_skd = mog_skd_config.create_model()
        logger.info(f"Using MoG-SKD with config: {mog_skd_config.to_dict()}")
    else:
        mog_skd = None
        logger.info("Using standard distillation")

    # Get template
    if args.dataset_name == 'anli':
        prompts = DatasetTemplates('anli', None)
    else:
        prompts = DatasetTemplates(
            f"{args.dataset_name}"
            if args.dataset_config_name is None
            else f"{args.dataset_name}/{args.dataset_config_name}"
        )
    template = prompts[args.template_name]

    # Preprocessing functions (simplified for brevity)
    def preprocess_train(examples):
        bs = len(examples[column_names[0]])
        input_texts = []
        target_texts = []
        for i in range(bs):
            ex = {k: examples[k][i] for k in column_names}
            input, target = template.apply(ex)
            input_texts.append(input)
            target_texts.append(target)

        model_inputs = tokenizer(
            input_texts,
            max_length=args.max_length,
            truncation=True,
            padding=False,
        )

        with tokenizer.as_target_tokenizer():
            tokenized_targets = tokenizer(
                target_texts,
                max_length=args.target_max_length,
                truncation=True,
                padding=False,
            )
            model_inputs['labels'] = [
                [(t if t != tokenizer.pad_token_id else -100) for t in targets]
                for targets in tokenized_targets["input_ids"]
            ]
        return model_inputs

    def preprocess_eval(examples):
        bs = len(examples[column_names[0]])
        input_texts = []
        target_texts = []
        answer_choices_texts = []
        for i in range(bs):
            ex = {k: examples[k][i] for k in column_names}
            input, target = template.apply(ex)
            ex_answer_choices = template.get_answer_choices_list(ex)
            input_texts.append(input)
            target_texts.append(target)
            answer_choices_texts.append(ex_answer_choices)

        tokenized_inputs = tokenizer(
            input_texts,
            max_length=args.max_length,
            truncation=True,
            padding=False,
        )

        tokenized_targets = [
            tokenizer(ans, padding=True, max_length=args.target_max_length, truncation=True)
            for ans in answer_choices_texts
        ]

        features = {
            k: [
                [elem for _ in range(len(tokenized_targets[idx]["input_ids"]))]
                for idx, elem in enumerate(v)
            ]
            for k, v in tokenized_inputs.items()
        }

        features["labels"] = [tokenized_targets[idx]["input_ids"] for idx in range(bs)]
        features["labels_attention_mask"] = [
            tokenized_targets[idx]["attention_mask"] for idx in range(bs)
        ]
        features["targets"] = [
            answer_choices_texts[idx].index(t) for idx, t in enumerate(target_texts)
        ]

        return features

    with accelerator.main_process_first():
        eval_dataset = raw_eval_dataset.map(preprocess_eval, batched=True, remove_columns=column_names)

        if args.num_shots is not None:
            sample_indices = random.sample(range(0, len(raw_train_dataset)), k=args.num_shots)
            raw_train_dataset = raw_train_dataset.select(sample_indices)
        train_dataset = raw_train_dataset.map(preprocess_train, batched=True, remove_columns=column_names)

    # DataLoaders
    train_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=train_collator,
        batch_size=args.per_device_train_batch_size
    )

    eval_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=eval_collator,
        batch_size=args.per_device_eval_batch_size
    )

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    # Add MoG-SKD parameters to optimizer
    if mog_skd is not None:
        optimizer_grouped_parameters.append({
            "params": mog_skd.parameters(),
            "lr": args.learning_rate * 0.1,  # Lower LR for gating
            "weight_decay": 0.0,
        })

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare for acceleration
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    if mog_skd is not None:
        mog_skd = accelerator.prepare(mog_skd)

    # Metric
    from datasets import load_metric
    metric = load_metric("")

    # Training loop
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size = {args.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation = {args.gradient_accumulation_steps}")
    logger.info(f"  Total steps = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    global_steps = 0
    maxscore = 0

    result_table = []
    mog_skd_logs_all = []

    for epoch in range(args.num_train_epochs):
        model.train()
        epoch_logs = []

        for step, batch in enumerate(train_dataloader):
            # Forward pass
            outputs = model(**batch)
            student_loss = outputs.loss
            student_logits = outputs.logits

            # Teacher forward pass (no grad)
            with torch.no_grad():
                teacher_outputs = teacher_model(**batch)
                teacher_logits = teacher_outputs.logits

            # Reshape logits for distillation
            # From [batch_size * seq_len, vocab_size] to [batch_size, vocab_size]
            # We use the mean over sequence for simplicity
            student_logits_flat = student_logits.view(-1, student_logits.size(-1))
            teacher_logits_flat = teacher_logits.view(-1, teacher_logits.size(-1))

            # Sample to reduce memory if needed
            if student_logits_flat.size(0) > 1000:
                indices = torch.randperm(student_logits_flat.size(0))[:1000]
                student_logits_flat = student_logits_flat[indices]
                teacher_logits_flat = teacher_logits_flat[indices]

            # Compute distillation loss
            if mog_skd is not None:
                distill_loss, logs = mog_skd(
                    student_logits_flat,
                    teacher_logits_flat,
                    return_details=True
                )
                epoch_logs.append(logs)
            else:
                # Standard KL divergence
                distill_loss = torch.nn.functional.kl_div(
                    torch.log_softmax(student_logits_flat / args.temperature, dim=-1),
                    torch.softmax(teacher_logits_flat / args.temperature, dim=-1),
                    reduction='batchmean'
                ) * (args.temperature ** 2)

            # Combined loss
            loss = student_loss + 0.5 * distill_loss
            loss = loss / args.gradient_accumulation_steps

            # Backward
            accelerator.backward(loss)

            if step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                global_steps += 1

            if global_steps >= args.max_train_steps:
                break

        # Aggregate logs for this epoch
        if mog_skd is not None and len(epoch_logs) > 0:
            epoch_stats = aggregate_logs(epoch_logs)
            mog_skd_logs_all.append(epoch_stats)

            logger.info(f"Epoch {epoch+1} MoG-SKD Stats:")
            logger.info(f"  Fisher: loss={epoch_stats['loss_fisher']:.4f}, weight={epoch_stats['weight_fisher']:.4f}")
            logger.info(f"  Euclid: loss={epoch_stats['loss_euclid']:.4f}, weight={epoch_stats['weight_euclid']:.4f}")
            logger.info(f"  Hyper:  loss={epoch_stats['loss_hyper']:.4f}, weight={epoch_stats['weight_hyper']:.4f}")
            logger.info(f"  Gating entropy: {epoch_stats['gating_entropy']:.4f}")

        # Evaluation
        model.eval()
        for batch in eval_dataloader:
            model_inputs = {k: batch[k] for k in ["input_ids", "attention_mask", "labels"]}
            with torch.no_grad():
                logits = model(**model_inputs).logits

            masked_log_probs = batch["labels_attention_mask"].unsqueeze(-1) * torch.log_softmax(logits, dim=-1)
            seq_token_log_probs = torch.gather(masked_log_probs, -1, batch["labels"].unsqueeze(-1))
            seq_log_prob = seq_token_log_probs.squeeze(dim=-1).sum(dim=-1)
            seq_log_prob = seq_log_prob.view(batch["targets"].size(0), -1)
            predictions = seq_log_prob.argmax(dim=-1)

            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["targets"]),
            )

        eval_metric = metric.compute()
        score = eval_metric["accuracy"]
        accelerator.print(f"Epoch {epoch+1} Accuracy: {score}")

        if score > maxscore:
            maxscore = score
            # Save best model
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(os.path.join(args.output_dir, "best_model"))

        result_table.append({
            "epoch": epoch + 1,
            "step": global_steps,
            "accuracy": score,
            "max_accuracy": maxscore,
        })

    # Save results
    if accelerator.is_main_process:
        # Save training results
        with open(os.path.join(args.output_dir, "results.csv"), "w") as f:
            writer = csv.DictWriter(f, fieldnames=result_table[0].keys())
            writer.writeheader()
            writer.writerows(result_table)

        # Save MoG-SKD logs
        if mog_skd is not None and len(mog_skd_logs_all) > 0:
            import json
            with open(os.path.join(args.output_dir, "mog_skd_logs.json"), "w") as f:
                json.dump(mog_skd_logs_all, f, indent=2)

    logger.info(f"Training complete. Max accuracy: {maxscore}")


def aggregate_logs(logs_list):
    """Aggregate logs from multiple batches."""
    aggregated = {}

    # Average all scalar fields
    for key in logs_list[0].keys():
        if key != 'per_sample_data':
            values = [log[key] for log in logs_list]
            aggregated[key] = sum(values) / len(values)

    return aggregated


if __name__ == "__main__":
    main()
