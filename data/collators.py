import torch


class BaseCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _pad_batch(self, batch, max_length):
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

    def prepare_batch(self, batch, max_length=None):
        # batch is a list of dicts, each containing "input_ids", "attention_mask", "labels", "images"
        # Sometimes batch can be a tuple (especially during validation), convert to list
        if isinstance(batch, tuple):
            batch = list(batch)

        # let's convert it to a dict of lists of tensors
        batch = {k: [item[k] for item in batch] for k in batch[0]}

        if max_length is not None:
            if self.is_validation:
                # During validation, truncate sequences instead of discarding them
                batch = self._truncate_samples_that_are_too_long(batch, max_length)
            else:
                # During training, discard samples that are too long
                batch = self._discard_samples_that_are_too_long(batch, max_length)

        # If all samples were filtered out during training, raise an error
        # During validation, this shouldn't happen since we truncate instead
        if not batch["input_ids"]:
            if self.is_validation:
                # This shouldn't happen with truncation, but handle gracefully
                return None
            else:
                raise ValueError(
                    "All samples in batch were filtered out. This shouldn't happen in VLM training."
                )

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
            "image_grids": batch.get("image_grids", []),
        }

    def _discard_samples_that_are_too_long(self, batch, max_length):
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
            # Return empty dict, not tuple
            return {"input_ids": [], "labels": [], "attention_mask": [], "images": []}
        batch_token_ids, batch_labels, batch_attentions, batch_images = zip(*filtered)
        return {
            "input_ids": list(batch_token_ids),
            "labels": list(batch_labels),
            "attention_mask": list(batch_attentions),
            "images": list(batch_images),
        }

    def _truncate_samples_that_are_too_long(self, batch, max_length):
        """Truncate sequences that are too long instead of discarding them (for validation)."""
        truncated_ids = []
        truncated_labels = []
        truncated_attentions = []

        for ids, label, attn in zip(
            batch["input_ids"], batch["labels"], batch["attention_mask"]
        ):
            if len(ids) > max_length:
                # Truncate from the end (keeping the beginning with image tokens)
                truncated_ids.append(ids[:max_length])
                truncated_labels.append(label[:max_length])
                truncated_attentions.append(attn[:max_length])
            else:
                truncated_ids.append(ids)
                truncated_labels.append(label)
                truncated_attentions.append(attn)

        return {
            "input_ids": truncated_ids,
            "labels": truncated_labels,
            "attention_mask": truncated_attentions,
            "images": batch["images"],
            "image_grids": batch.get("image_grids", []),
        }


class VQACollator(BaseCollator):  # Visual Question Answering Collator
    def __init__(self, tokenizer, max_length, is_validation=False):
        self.max_length = max_length
        self.is_validation = is_validation
        super().__init__(tokenizer)

    def _pad_batch(
        self, batch, max_length
    ):  # Use -100 as the pad value for labels, so that it's ignored by the loss
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

    def __call__(self, batch):
        batch = self.prepare_batch(batch, max_length=self.max_length)
        return batch
