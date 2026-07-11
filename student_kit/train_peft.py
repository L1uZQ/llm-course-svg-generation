"""
train_peft.py — LoRA fine-tuning script for Gemma 3 270M
=========================================================
用LoRA微调Gemma 3 270M，在train.jsonl上训练，损失只在SVG部分计算。

用法:
    python student_kit/train_peft.py [train.jsonl] [valid.jsonl]

超参数可通过命令行参数调整，或直接修改DEFAULT_CONFIG。
"""

import json
import os
import sys
import argparse
import math
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)
from tqdm import tqdm

# ─── 默认配置 ─────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # 模型
    "model_dir": "gemma3-270m",

    # LoRA
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],

    # 训练
    "learning_rate": 2e-4,
    "num_epochs": 5,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,

    # 数据 (87%样本在1536以内，适配8GB显存)
    "max_length": 1536,        # 序列最大长度
    "max_svg_tokens": 1024,    # SVG部分最大token数

    # 保存
    "output_dir": "adapter",
    "save_steps": 50,
    "eval_steps": 50,
    "logging_steps": 10,

    # 早停
    "early_stopping_patience": 3,

    # 混合精度 (RTX3060用fp16更稳定)
    "bf16": False,
    "fp16": True,
}


# ─── 数据集 ────────────────────────────────────────────────────────────────

class SVGDataset(Dataset):
    """SVG徽标数据集，使用Gemma 3格式自动mask提示词部分的loss
    
    Gemma 3格式:
    <bos><start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n<start_of_turn>model\n{svg}<end_of_turn><eos>
    """

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 2560,
                 max_svg_tokens: int = 1536):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_svg_tokens = max_svg_tokens

        self.examples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                self.examples.append(item["messages"])

    @staticmethod
    def _format_gemma3(messages, include_assistant=True):
        """将messages格式化为Gemma 3的对话文本"""
        system = messages[0]["content"]
        user = messages[1]["content"]

        # 构建prompt部分
        prompt = (
            f"<bos><start_of_turn>user\n"
            f"{system}\n\n{user}"
            f"<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )

        if include_assistant and len(messages) > 2:
            assistant = messages[2]["content"]
            full = f"{prompt}{assistant}<end_of_turn><eos>"
            return full, prompt
        return prompt, prompt

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        messages = self.examples[idx]

        # 使用Gemma 3格式构建文本
        full_text, prompt_text = self._format_gemma3(messages)

        # Tokenize全序列
        full_encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        input_ids = full_encoding["input_ids"][0]
        attention_mask = full_encoding["attention_mask"][0]

        # Tokenize prompt部分以定位SVG起始
        prompt_encoding = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        prompt_len = prompt_encoding["input_ids"].shape[1]

        # 构建labels: 全部先设为-100，只对SVG部分计算loss
        labels = torch.full_like(input_ids, -100)
        svg_start = min(prompt_len, len(input_ids))
        svg_end = min(svg_start + self.max_svg_tokens, len(input_ids))
        labels[svg_start:svg_end] = input_ids[svg_start:svg_end].clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class DataCollator:
    """简单DataCollator: padding + stacking"""

    def __init__(self, tokenizer, pad_token_id):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len = max(item["input_ids"].shape[0] for item in batch)

        input_ids = []
        attention_masks = []
        labels = []

        for item in batch:
            pad_len = max_len - item["input_ids"].shape[0]
            input_ids.append(torch.cat([
                item["input_ids"],
                torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
            ]))
            attention_masks.append(torch.cat([
                item["attention_mask"],
                torch.zeros(pad_len, dtype=torch.long),
            ]))
            labels.append(torch.cat([
                item["labels"],
                torch.full((pad_len,), -100, dtype=torch.long),
            ]))

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels),
        }


# ─── 训练逻辑 ──────────────────────────────────────────────────────────────

def compute_loss(model, batch):
    """前向传播计算loss"""
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    return outputs.loss


def evaluate(model, dataloader, device):
    """在验证集上评估"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = compute_loss(model, batch)
            total_loss += loss.item()
            num_batches += 1

    model.train()
    return total_loss / max(num_batches, 1)


def train(config: dict):
    """主训练函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    print(f"Config: {json.dumps(config, indent=2, ensure_ascii=False)}")

    # 1. 加载tokenizer
    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_dir"], trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载基座模型
    print("[2/5] Loading base model...")
    dtype = torch.bfloat16 if config["bf16"] else (
        torch.float16 if config["fp16"] else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["model_dir"],
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False  # 训练时禁用KV cache
    model.gradient_checkpointing_enable()  # 节省显存

    # 3. 应用LoRA
    print("[3/5] Applying LoRA...")
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. 加载数据
    print("[4/5] Loading datasets...")
    train_dataset = SVGDataset(
        config["train_data"],
        tokenizer,
        max_length=config["max_length"],
        max_svg_tokens=config["max_svg_tokens"],
    )
    valid_dataset = SVGDataset(
        config["valid_data"],
        tokenizer,
        max_length=config["max_length"],
        max_svg_tokens=config["max_svg_tokens"],
    )

    collator = DataCollator(tokenizer, tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collator,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collator,
    )

    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Valid: {len(valid_dataset)} samples, {len(valid_loader)} batches")

    # 5. 优化器 & 调度器
    print("[5/5] Setting up optimizer & scheduler...")
    total_steps = len(train_loader) * config["num_epochs"] // config["gradient_accumulation_steps"]
    warmup_steps = int(total_steps * config["warmup_ratio"])

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ─── 训练循环 ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Starting training...")
    print(f"Total steps: {total_steps}, Warmup: {warmup_steps}")
    print(f"Accumulation: {config['gradient_accumulation_steps']}")
    print("=" * 60 + "\n")

    global_step = 0
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []
    log_history = []

    for epoch in range(config["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")

        for step, batch in enumerate(progress):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = compute_loss(model, batch)
            loss = loss / config["gradient_accumulation_steps"]
            loss.backward()

            epoch_loss += loss.item()

            if (step + 1) % config["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                train_losses.append(epoch_loss / (step + 1))

                # 日志
                if global_step % config["logging_steps"] == 0:
                    lr = scheduler.get_last_lr()[0]
                    progress.set_postfix({
                        "loss": f"{train_losses[-1]:.4f}",
                        "lr": f"{lr:.2e}",
                    })

                # 评估 & 保存
                if global_step % config["eval_steps"] == 0:
                    val_loss = evaluate(model, valid_loader, device)
                    val_losses.append(val_loss)
                    log_history.append({
                        "step": global_step,
                        "epoch": epoch + 1,
                        "train_loss": round(train_losses[-1], 6),
                        "val_loss": round(val_loss, 6),
                        "lr": scheduler.get_last_lr()[0],
                    })

                    tqdm.write(
                        f"  Step {global_step}: train_loss={train_losses[-1]:.4f}, "
                        f"val_loss={val_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}"
                    )

                    # 早停检查
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # 保存最佳适配器
                        model.save_pretrained(config["output_dir"])
                        tqdm.write(f"  -> Best model saved (val_loss={val_loss:.4f})")
                    else:
                        patience_counter += 1
                        if patience_counter >= config["early_stopping_patience"]:
                            tqdm.write(f"  Early stopping at step {global_step}")
                            break

            # 也非accumulation步也可以保存
            if (step + 1) % (config["save_steps"] * config["gradient_accumulation_steps"]) == 0:
                checkpoint_dir = f"{config['output_dir']}_checkpoint_{global_step}"
                model.save_pretrained(checkpoint_dir)

        if patience_counter >= config["early_stopping_patience"]:
            break

    # ─── 最终保存 ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best val_loss: {best_val_loss:.4f}")

    # 保存最终适配器（如果提前停止了，之前已保存最佳）
    model.save_pretrained(config["output_dir"])

    # 保存训练日志
    log_path = os.path.join(config["output_dir"], "train_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": config,
            "history": log_history,
            "best_val_loss": best_val_loss,
            "final_step": global_step,
        }, f, ensure_ascii=False, indent=2)

    print(f"Adapter saved to: {config['output_dir']}")
    print(f"Training log saved to: {log_path}")

    # 返回训练结果用于报告
    return {
        "best_val_loss": best_val_loss,
        "final_step": global_step,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "log_history": log_history,
    }


# ─── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune Gemma 3 270M for SVG logo generation")
    parser.add_argument("train_data", nargs="?", default="data_repo/train.jsonl",
                        help="Path to train.jsonl")
    parser.add_argument("valid_data", nargs="?", default="data_repo/valid.jsonl",
                        help="Path to valid.jsonl")
    parser.add_argument("--model-dir", default="gemma3-270m",
                        help="Path to base model")
    parser.add_argument("--output-dir", default="adapter",
                        help="Output directory for adapter")
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size per device")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Max sequence length")
    parser.add_argument("--max-svg-tokens", type=int, default=1536,
                        help="Max SVG tokens for loss")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience")
    parser.add_argument("--fp16", action="store_true",
                        help="Use fp16 (default: bf16)")
    args = parser.parse_args()

    # 构建配置
    config = dict(DEFAULT_CONFIG)
    config.update({
        "train_data": args.train_data,
        "valid_data": args.valid_data,
        "model_dir": args.model_dir,
        "output_dir": args.output_dir,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "learning_rate": args.lr,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "max_length": args.max_length,
        "max_svg_tokens": args.max_svg_tokens,
        "early_stopping_patience": args.patience,
        "bf16": not args.fp16,
        "fp16": args.fp16,
    })

    train(config)


if __name__ == "__main__":
    main()
