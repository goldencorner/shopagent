"""训练数据构造的单元测试（用假 tokenizer，不下载模型）。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("datasets")

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "train_router_lora", ROOT / "scripts" / "13_train_router_lora.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeTokenizer:
    """只实现 build_dataset 用到的两个方法，字符级切 token。"""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [ord(ch) % 97 + 1 for ch in text]
        if truncation and max_length:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def test_build_dataset_masks_prompt_and_keeps_answer():
    rows = [
        {"query": "推荐一款露营帐篷", "intent": "shop", "en_query": "camping tent"},
        {"query": "你好", "intent": "chitchat", "en_query": ""},
    ]
    ds = MODULE.build_dataset(rows, FakeTokenizer(), max_len=512)
    assert len(ds) == 2
    labels = ds[0]["labels"]
    first_trainable = next(i for i, token in enumerate(labels) if token != -100)
    assert first_trainable > 0
    assert all(token == -100 for token in labels[:first_trainable])
    assert labels[first_trainable:] == ds[0]["input_ids"][first_trainable:]


def test_build_dataset_drops_rows_fully_truncated_by_prompt():
    rows = [{"query": "推荐一款露营帐篷", "intent": "shop", "en_query": "camping tent"}]
    with pytest.raises(SystemExit):
        MODULE.build_dataset(rows, FakeTokenizer(), max_len=5)


def test_pad_collator_keeps_prompt_mask_and_pads_labels_with_ignore_index():
    torch = pytest.importorskip("torch")
    collator = MODULE.PadCollator(pad_id=0)
    batch = collator(
        [
            {"input_ids": [5, -100, 7], "attention_mask": [1, 1, 1], "labels": [-100, -100, 7]},
            {"input_ids": [5, -100], "attention_mask": [1, 1], "labels": [-100, -100]},
        ]
    )
    assert batch["input_ids"].tolist() == [[5, -100, 7], [5, -100, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert batch["labels"].tolist() == [[-100, -100, 7], [-100, -100, -100]]
    assert batch["input_ids"].dtype == torch.long
