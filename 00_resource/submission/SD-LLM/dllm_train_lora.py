import argparse
import os
import json
from typing import Optional, List, Dict, Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
    TrainerCallback,
)
import sys, torch.distributed as dist
from peft import LoraConfig, get_peft_model

from dllm_train_dataset_collator import SVGDataset, CausalLMCollator

from rank0_utils import patch_print, configure_logging_for_rank0
patch_print()                     # built-in print becomes no-op on nonzero ranks
configure_logging_for_rank0()     # quiets library loggers on nonzero ranks

class ConsoleLossCallback(TrainerCallback):
    '''
    Custom callback prints and optionally logs training loss and learning-rate updates to the console (and a file).
    '''
    def __init__(self, log_file: str = "") -> None:
        super().__init__()
        self.log_file = log_file
        self._fh = None
        if self.log_file:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            self._fh = open(self.log_file, "a", encoding="utf-8")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = state.global_step
        loss = logs.get("loss", logs.get("train_loss"))
        lr = logs.get("learning_rate")
        msg = f"step={step}"
        if loss is not None:
            msg += f" | loss={loss:.6f}"
        if lr is not None:
            msg += f" | lr={lr:.6e}"
        print(msg, flush=True)
        if self._fh is not None:
            self._fh.write(msg + "\n")
            self._fh.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass

    
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Qwen3 (last-assistant loss only)")

    # Data & model
    parser.add_argument("--train_file", type=str, default="/root/autodl-tmp/merged_for_training_prompted.json", help="Path to JSON/JSONL dataset")
    parser.add_argument("--model_name_or_path", type=str, default="/root/autodl-tmp/qwen3-4b-instruct_lora_v1", help="Base model to fine-tune")
    parser.add_argument("--output_dir", type=str, default="./qwen3_4b_instruct_lora_v1_last_assistant")

    # Sequence & tokenizer
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--local_files_only", action="store_true", help="Load tokenizer/model only from local cache")

    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")

    # Precision & memory (no quantization; casting dtypes is not quantization)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    # Dataloader & seed
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    # Logging
    parser.add_argument("--log_file", type=str, default="", help="Optional file to append plain logs")

    # Tools injection (global tools for samples missing tools in data)
    parser.add_argument("--tools_file", type=str, default="", help="Path to JSON file with a top-level list of tools (OpenAI/Qwen schema)")
    parser.add_argument("--tools_json", type=str, default="", help="Inline JSON string representing a list of tools")

    # LoRA config
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # attention method
    parser.add_argument("--attn_impl", type=str, default="sdpa", choices=["flash_attention_2", "sdpa", "eager"],)

    return parser.parse_args()


def build_lora_model(base_model, args: argparse.Namespace):
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    lora_model = get_peft_model(base_model, lora_cfg)
    lora_model.print_trainable_parameters()
    return lora_model


def main():

    import torch
    from torch.backends.cuda import sdp_kernel
    # Sanity: are flash kernels compiled for this GPU + dtype?
    print("flash_sdp_enabled():", torch.backends.cuda.flash_sdp_enabled())          # True expected
    print("mem_efficient_sdp_enabled():", torch.backends.cuda.mem_efficient_sdp_enabled())
    print("math_sdp_enabled():", torch.backends.cuda.math_sdp_enabled())
    # Force SDPA to use the Flash path only (error/fallback if unsupported)
    sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False)
    
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[DEBUG]Loading tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"[DEBUG]Loading base model: {args.model_name_or_path}")
    # Optional precision cast (no quantization)
    # NOTE: prefer passing dtype at load (safer than post-hoc .to(dtype=...))
    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        torch_dtype=torch_dtype,  # dtype cast on load
        attn_implementation=args.attn_impl,
    )
    
    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            print('[DEBUG] Not using reentrant')
            model.gradient_checkpointing_enable()
        # disable kv cache
        if hasattr(model, "config"):
            model.config.use_cache = False

    # Wrap with LoRA adapters
    model = build_lora_model(model, args)  # get_peft_model(base_model, lora_cfg)

    # Dataset
    print(f"[DEBUG]Loading dataset: {args.train_file}")
    # ========== train_dataset, DataLoader, datacollator =======================
    # expect torch.utils.data.Dataset, must implement __len__(), __getitem__(index)
    # for llm task, __getitem__(index) should return a dict of {'input_ids', 'attention_mask', 'labels'}, the value can be pytorch tensors or numpy arrays
    train_dataset = SVGDataset(args.train_file, tokenizer, args.max_seq_length)
    data_collator = CausalLMCollator(tokenizer=tokenizer, pad_to_multiple_of=8)
    callbacks = [ConsoleLossCallback(args.log_file)] if args.log_file else None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        lr_scheduler_type=args.lr_scheduler_type,
        optim="adamw_torch",
        bf16=args.bf16,
        fp16=args.fp16 and not args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=[],
        remove_unused_columns=False,
        seed=args.seed,
        save_safetensors=True,

        # DDP
        ddp_backend="nccl",
        ddp_find_unused_parameters=False, # important with LoRA + checkpointing
    )

    

    trainer = Trainer(
        # model = get_peft_model(base_model, lora_cfg) + gradient checkpointing
        # Trainer() will call output = model(**batch) during training, output should be a dict contains at least ['loss']
        model=model,

        # all training parameters
        args=training_args,
        tokenizer=tokenizer,

        # ========== train_dataset, DataLoader, datacollator =======================
        # user provides train_dataset and data_collator, Trainer() provides DataLoader
        # train_dataset provide samples one by one(__getitem__()), DataLoader turns them into list[dict], data_collator handle padding and stack the list[dict] into a rectangular torch.Tensor (dict{str:tensor})
        train_dataset=train_dataset,

        # expect a callable, take list[dict] -> dict[str, torch.Tensor], which is what the model actually receives
        # collator will find the longest seq in the batch and pad all seq to the same length, then stack them to tensor
        # data_collator = None if your train_dataset is already fix length and tensorized.
        data_collator=data_collator,

        # expect list[TrainerCallback]
        callbacks=callbacks,
    )
        
    trainer.train()

    # Save adapter
    trainer.save_state()  # save metadata to trainer_state.json for resume training
    trainer.save_model(args.output_dir)  # save model weights/config/tokenizer

    print("Training complete. Adapter saved to:", args.output_dir)


if __name__ == "__main__":
    main()
