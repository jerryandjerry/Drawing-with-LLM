from __future__ import annotations
import json
from typing import Any, Dict, List, Optional
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and all(k in obj for k in ("system", "user", "assistant")):
                data.append(obj)
    if not data:
        raise ValueError("No valid records with keys: 'system', 'user', 'assistant'.")
    return data


class SVGDataset(Dataset):
    """
    Returns dict with:
      - input_ids: LongTensor[T]
      - attention_mask: LongTensor[T]  (all 1s here; collator pads)
      - labels: LongTensor[T]          (loss only on assistant tokens; others -100)
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
    ) -> None:
        super().__init__()
        self.tok = tokenizer
        self.max_len = int(max_seq_length)

        # Ensure padding config
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"

        self.samples = _load_jsonl(data_path)

    def __len__(self) -> int:
        return len(self.samples)

    def _tmpl(self, messages: List[Dict[str, str]]) -> List[int]:
        # Qwen3 chat template; supervised fine-tuning => no generation prompt
        return self.tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.samples[idx]
        sys_txt: str = rec.get("system", "")
        usr_txt: str = rec["user"]
        asst_txt: str = rec["assistant"]

        msgs: List[Dict[str, str]] = []
        if isinstance(sys_txt, str) and len(sys_txt) > 0:
            msgs.append({"role": "system", "content": sys_txt})
        msgs.append({"role": "user", "content": usr_txt})
        msgs.append({"role": "assistant", "content": asst_txt})

        # Encode full conversation (system + user + assistant)
        full_ids: List[int] = self._tmpl(msgs)
        total_len = len(full_ids)

        # Encode prefix (system + user); assistant span is [start, total_len)
        prefix_msgs = [m for m in msgs if m["role"] != "assistant"]
        prefix_ids: List[int] = self._tmpl(prefix_msgs)
        start = len(prefix_ids)

        # Left-truncate to fit max_seq_length (keep most recent)
        if total_len > self.max_len:
            cut = total_len - self.max_len
            full_ids = full_ids[cut:]
            start = max(0, start - cut)
            total_len = self.max_len

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        if start > 0:
            labels[:start] = -100  # ignore system+user in loss

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class CausalLMCollator:
    """
    Pads to the longest example in the batch:
      - input_ids -> pad_token_id
      - attention_mask -> 0
      - labels -> -100
    Optionally pad to a multiple (e.g., 8) for better throughput on some hardware.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, pad_to_multiple_of: Optional[int] = None) -> None:
        self.tok = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        ids  = [f["input_ids"] for f in features]
        ams  = [f["attention_mask"] for f in features]
        labs = [f["labels"] for f in features]

        pad_id = self.tok.pad_token_id

        batch_input_ids = torch.nn.utils.rnn.pad_sequence(ids,  batch_first=True, padding_value=pad_id)
        batch_attention = torch.nn.utils.rnn.pad_sequence(ams,  batch_first=True, padding_value=0)
        batch_labels    = torch.nn.utils.rnn.pad_sequence(labs, batch_first=True, padding_value=-100)

        if self.pad_to_multiple_of is not None:
            def _pad_to_multiple(t: torch.Tensor, value: int) -> torch.Tensor:
                L = t.size(1)
                need = (self.pad_to_multiple_of - (L % self.pad_to_multiple_of)) % self.pad_to_multiple_of
                if need == 0:
                    return t
                extra = torch.full((t.size(0), need), value, dtype=t.dtype, device=t.device)
                return torch.cat([t, extra], dim=1)

            batch_input_ids = _pad_to_multiple(batch_input_ids, pad_id)
            batch_attention = _pad_to_multiple(batch_attention, 0)
            batch_labels    = _pad_to_multiple(batch_labels, -100)

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention,
            "labels": batch_labels,
        }

