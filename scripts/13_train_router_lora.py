"""用 LoRA 微调 Qwen2.5-1.5B-Instruct，让它接管意图分类 + 中文→英文检索词。

在 AutoDL 4090 上跑，约 40 分钟（含装环境和下模型），费用 3 元以内。
训练数据来自 scripts/12_make_router_data.py。

AutoDL 上建议的命令（示例，按镜像里的 torch 版本调整）:
    pip install -U transformers peft datasets accelerate
    export HF_ENDPOINT=https://hf-mirror.com
    python scripts/13_train_router_lora.py --epochs 3 --merge

本地只验证数据/脚本，不训练:
    python scripts/13_train_router_lora.py --smoke --allow-cpu

依赖: torch transformers peft datasets accelerate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shopagent.router_prompt import LOCAL_ROUTER_PROMPT  # noqa: E402

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_rows(path: Path, limit: int | None = None) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"找不到数据文件：{path}，先跑 scripts/12_make_router_data.py")
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
        if limit and len(rows) >= limit:
            break
    return rows


def build_dataset(rows: list[dict], tokenizer, max_len: int):
    from datasets import Dataset

    records = []
    skipped = 0
    for row in rows:
        prompt_messages = [
            {
                "role": "user",
                "content": LOCAL_ROUTER_PROMPT.format(query=str(row["query"])[:500]),
            }
        ]
        answer = json.dumps(
            {"intent": row["intent"], "en_query": row.get("en_query", "")},
            ensure_ascii=False,
        )
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            prompt_messages + [{"role": "assistant", "content": answer}],
            tokenize=False,
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_len,
        )
        input_ids = full["input_ids"]
        labels = list(input_ids)
        n_prompt = min(len(prompt_ids), len(input_ids))
        labels[:n_prompt] = [-100] * n_prompt
        if all(label == -100 for label in labels):
            skipped += 1
            continue
        records.append(
            {
                "input_ids": input_ids,
                "attention_mask": full["attention_mask"],
                "labels": labels,
            }
        )
    if skipped:
        print(f"跳过 {skipped} 条（截断后没有可训练的答案 token）")
    if not records:
        raise SystemExit("没有可用训练样本")
    return Dataset.from_list(records)


class PadCollator:
    """按 batch 内最长样本右侧 padding，labels 用 -100 填充。

    不能用 transformers 的 DataCollatorForLanguageModeling：5.x 会忽略我们预计算的
    labels，直接从 input_ids 重建，导致 prompt 部分也参与训练（相当于没做 mask）。
    """

    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, features: list[dict]) -> dict:
        import torch

        max_len = max(len(item["input_ids"]) for item in features)
        input_ids, attention_mask, labels = [], [], []
        for item in features:
            pad = max_len - len(item["input_ids"])
            input_ids.append(list(item["input_ids"]) + [self.pad_id] * pad)
            attention_mask.append(list(item["attention_mask"]) + [0] * pad)
            labels.append(list(item["labels"]) + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_causal_lm(cls, name: str, dtype):
    """transformers 5.x 把 torch_dtype 改名成 dtype，这里兼容两代。"""
    try:
        return cls.from_pretrained(name, dtype=dtype, trust_remote_code=True)
    except TypeError:
        return cls.from_pretrained(name, torch_dtype=dtype, trust_remote_code=True)


def make_args(**kwargs):
    """兼容 transformers 新旧版本的参数名。"""
    from transformers import TrainingArguments

    try:
        return TrainingArguments(**kwargs)
    except TypeError:
        if "eval_strategy" in kwargs:
            kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
        return TrainingArguments(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "router"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output", default=str(ROOT / "data" / "router" / "lora_out"))
    parser.add_argument("--merged-dir", default=str(ROOT / "data" / "router" / "merged"))
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--smoke", action="store_true", help="只跑 200 条 × 1 epoch，验证环境")
    parser.add_argument("--dry-run", action="store_true", help="只构建数据集并打印样例，不训练")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--merge", action="store_true", help="训练结束后顺带合并 adapter")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--grad-checkpoint", action="store_true")
    args = parser.parse_args()

    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
        )
    except ImportError as exc:
        raise SystemExit(
            f"缺少训练依赖（{exc.name}）。先执行：\n"
            "  pip install torch transformers peft datasets accelerate"
        )

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)

    if args.merge_only:
        merge(args.model, output_dir, Path(args.merged_dir), AutoModelForCausalLM, PeftModel, AutoTokenizer)
        return

    if args.dry_run:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        rows = load_rows(data_dir / "train.jsonl", limit=20)
        ds = build_dataset(rows, tokenizer, args.max_len)
        print(ds)
        answer_ids = [token for token in ds[0]["labels"] if token != -100]
        print("第一条样本的答案部分：", tokenizer.decode(answer_ids))
        return

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("没有检测到 CUDA。AutoDL 请选 GPU 镜像；本地验证数据请加 --allow-cpu。")

    train_rows = load_rows(data_dir / "train.jsonl", limit=200 if args.smoke else None)
    valid_rows = load_rows(data_dir / "valid.jsonl", limit=50 if args.smoke else None)
    if args.smoke:
        output_dir = output_dir.parent / "lora_smoke"
        args.epochs = 1
        args.batch = min(args.batch, 2)
        args.grad_accum = 1
        print(f"smoke 模式：train={len(train_rows)} valid={len(valid_rows)} output={output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)
    model = load_causal_lm(AutoModelForCausalLM, args.model, dtype)
    if torch.cuda.is_available():
        model.to("cuda")
    model.config.use_cache = False
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        # LoRA 只训练 adapter，基座输入默认不需要梯度；不开这个 checkpointing 不会生效
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = build_dataset(train_rows, tokenizer, args.max_len)
    valid_ds = build_dataset(valid_rows, tokenizer, args.max_len)
    collator = PadCollator(tokenizer.pad_token_id)

    training_args = make_args(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch",
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
        dataloader_pin_memory=False,
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
    )
    try:
        trainer = Trainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = Trainer(tokenizer=tokenizer, **trainer_kwargs)

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    meta = {
        "base_model": args.model,
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "epochs": args.epochs,
        "lr": args.lr,
        "batch": args.batch,
        "grad_accum": args.grad_accum,
        "max_len": args.max_len,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": TARGET_MODULES,
        "final_metrics": trainer.state.log_history[-1] if trainer.state.log_history else {},
    }
    (output_dir / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"LoRA adapter 已保存：{output_dir}")

    if args.merge:
        merge(args.model, output_dir, Path(args.merged_dir), AutoModelForCausalLM, PeftModel, AutoTokenizer)


def merge(base_model, adapter_dir, merged_dir, AutoModelForCausalLM, PeftModel, AutoTokenizer):
    import torch

    if not (adapter_dir / "adapter_config.json").exists():
        raise SystemExit(f"{adapter_dir} 不是 LoRA adapter 目录")
    print(f"合并 {adapter_dir} → {merged_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    model = load_causal_lm(AutoModelForCausalLM, base_model, torch.float16)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.merge_and_unload()
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))
    print(f"合并完成：{merged_dir}")
    print("下一步转 GGUF（在 llama.cpp 目录）：")
    print(f"  python convert_hf_to_gguf.py {merged_dir} --outfile router-q8.gguf --outtype q8_0")


if __name__ == "__main__":
    main()
