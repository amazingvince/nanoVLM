from typing import Any, Dict, List, Optional

import torch
from transformers import PreTrainedTokenizer


class BaseCollator(object):
    """Base collator for batching and padding sequences.

    :param tokenizer: Tokenizer for padding operations
    """

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def _pad_batch(self, batch: Dict[str, List], max_length: int) -> None:
        """Pad batch items to uniform length.

        :param batch: Dictionary containing input_ids, labels, attention_mask
        :param max_length: Target length for padding
        """
        batch["input_ids"] = [
            torch.nn.functional.pad(
                ids, (max_length - len(ids), 0), value=self.tokenizer.pad_token_id
            )
            for ids in batch["input_ids"]
        ]
        batch["labels"] = [
            torch.nn.functional.pad(
                labels, (max_length - len(labels), 0), value=self.tokenizer.pad_token_id
            )
            for labels in batch["labels"]
        ]
        batch["attention_mask"] = [
            torch.nn.functional.pad(
                attention_mask, (max_length - len(attention_mask), 0), value=0
            )
            for attention_mask in batch["attention_mask"]
        ]

    def prepare_batch(
        self, batch: List[Dict[str, Any]], max_length: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Prepare batch by filtering, padding, and stacking tensors.

        :param batch: List of sample dictionaries
        :param max_length: Optional maximum sequence length
        :return: Dictionary with stacked tensors ready for model input
        """
        # 1) Handle empty
        if not batch:
            return {"input_ids": [], "labels": [], "attention_mask": [], "images": []}

        # 2) Drop None rows
        batch = [s for s in batch if s is not None]
        if not batch:
            return {"input_ids": [], "labels": [], "attention_mask": [], "images": []}

        # batch is a list of dicts, each containing "input_ids", "attention_mask", "labels", "images"
        # let's convert it to a dict of lists of tensors
        batch = {k: [item[k] for item in batch] for k in batch[0]}

        if max_length is not None:
            batch = self._discard_samples_that_are_too_long(batch, max_length)

        if len(batch["input_ids"]) == 0:
            return batch

        # Pad samples to max length
        if max_length is not None:
            max_len = max_length
        else:
            max_len = max(map(len, batch["input_ids"]))
        self._pad_batch(
            batch, max_len
        )  #  dictionaries in Python are mutable and passed by reference

        return {
            "input_ids": torch.stack(batch["input_ids"]),
            "attention_mask": torch.stack(batch["attention_mask"]),
            "images": batch["images"],
            "labels": torch.stack(batch["labels"]),
        }

    def _discard_samples_that_are_too_long(
        self, batch: Dict[str, List], max_length: int
    ) -> Dict[str, List]:
        """Filter out samples exceeding maximum length.

        :param batch: Batch dictionary with lists of tensors
        :param max_length: Maximum allowed sequence length
        :return: Filtered batch dictionary
        """
        filtered = [
            (ids, label, attn, img)
            for ids, label, attn, img in zip(
                batch["input_ids"],
                batch["labels"],
                batch["attention_mask"],
                batch["images"],
            )
            if len(ids) <= max_length
        ]
        if not filtered:
            return {"input_ids": [], "labels": [], "attention_mask": [], "images": []}
        batch_token_ids, batch_labels, batch_attentions, batch_images = zip(*filtered)
        return {
            "input_ids": list(batch_token_ids),
            "labels": list(batch_labels),
            "attention_mask": list(batch_attentions),
            "images": list(batch_images),
        }


class VQACollator(BaseCollator):  # Visual Question Answering Collator
    """Collator for visual question answering tasks with special label padding.

    :param tokenizer: Tokenizer for padding operations
    :param max_length: Maximum sequence length for padding
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int):
        self.max_length = max_length
        super().__init__(tokenizer)

    def _pad_batch(
        self, batch: Dict[str, List], max_length: int
    ) -> None:  # Reimplementing to use -100 as the pad value for labels, so that it's ignored by the loss
        """Pad batch with -100 for labels to ignore in loss computation.

        :param batch: Batch dictionary to pad
        :param max_length: Target padding length
        """
        batch["input_ids"] = [
            torch.nn.functional.pad(
                ids, (max_length - len(ids), 0), value=self.tokenizer.pad_token_id
            )
            for ids in batch["input_ids"]
        ]
        batch["labels"] = [
            torch.nn.functional.pad(labels, (max_length - len(labels), 0), value=-100)
            for labels in batch["labels"]
        ]
        batch["attention_mask"] = [
            torch.nn.functional.pad(
                attention_mask, (max_length - len(attention_mask), 0), value=0
            )
            for attention_mask in batch["attention_mask"]
        ]

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Process batch for DataLoader.

        :param batch: List of sample dictionaries
        :return: Collated batch dictionary
        """
        batch = self.prepare_batch(batch, max_length=self.max_length)
        return batch
