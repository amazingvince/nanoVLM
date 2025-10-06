import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from data.processors import get_image_string


class BaseDataset(Dataset):
    """Base dataset class for vision-language tasks with quality filtering.

    :param dataset: Source dataset to wrap
    :param tokenizer: Tokenizer for text processing
    :param image_processor: Image preprocessing pipeline
    :param mp_image_token_length: Number of tokens per image patch
    :param *_min_rating: Minimum quality ratings for filtering samples
    """

    def __init__(
        self,
        dataset: Any,
        tokenizer: PreTrainedTokenizer,
        image_processor: Any,
        mp_image_token_length: int,
        relevance_min_rating: int = 1,
        image_correspondence_min_rating: int = 1,
        visual_dependency_min_rating: int = 1,
        formatting_min_rating: int = 1,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length
        self.relevance_min_rating = relevance_min_rating
        self.image_correspondence_min_rating = image_correspondence_min_rating
        self.visual_dependency_min_rating = visual_dependency_min_rating
        self.formatting_min_rating = formatting_min_rating
        self.prefix_len = self._get_prefix_len()

    def __len__(self) -> int:
        return len(self.dataset)

    def _get_prefix_len(self) -> int:
        """Calculate prefix length for assistant responses in chat template.

        :return: Number of tokens in assistant response prefix
        """
        random_string_5_letters = "xzyvd"
        random_string_chat_templated = self.tokenizer.apply_chat_template(
            [{"role": "assistant", "content": random_string_5_letters}],
            tokenize=False,
            add_special_tokens=False,
        )
        random_string_location = random_string_chat_templated.find(
            random_string_5_letters
        )
        return len(
            self.tokenizer.encode(random_string_chat_templated[:random_string_location])
        )

    def _get_messages(
        self, item: Dict[str, Any], splitted_image_counts: List[Tuple[int, int]]
    ) -> List[Dict[str, str]]:
        """Extract and filter messages from dataset item based on quality ratings.

        :param item: Dataset item containing texts and ratings
        :param splitted_image_counts: List of (height, width) split counts
        :return: List of message dictionaries with role and content
        """
        messages = []
        for index, text in enumerate(item["texts"]):
            try:
                if (
                    item.get("relevance_ratings") is not None
                    and item["relevance_ratings"][index] is not None
                    and item["relevance_ratings"][index] < self.relevance_min_rating
                ):
                    continue
                if (
                    item.get("image_correspondence_ratings") is not None
                    and item["image_correspondence_ratings"][index] is not None
                    and item["image_correspondence_ratings"][index]
                    < self.image_correspondence_min_rating
                ):
                    continue
                if (
                    item.get("visual_dependency_ratings") is not None
                    and item["visual_dependency_ratings"][index] is not None
                    and item["visual_dependency_ratings"][index]
                    < self.visual_dependency_min_rating
                ):
                    continue
                if (
                    item.get("formatting_ratings") is not None
                    and item["formatting_ratings"][index] is not None
                    and item["formatting_ratings"][index] < self.formatting_min_rating
                ):
                    continue
            except Exception as e:
                logging.warning(f"Error processing item: {item}, index: {index}: {e}")

            messages.append({"role": "user", "content": text["user"]})
            messages.append({"role": "assistant", "content": text["assistant"]})

        if len(messages) == 0:
            return messages

        # Safety check to ensure no image tokens are present in the text before adding them.
        for msg in messages:
            if self.tokenizer.image_token in msg["content"]:
                logging.warning(
                    f"Found and removed an image token in the {msg['role']} text before adding the image string."
                )
                msg["content"] = msg["content"].replace(self.tokenizer.image_token, "")

        if len(splitted_image_counts) > 0:
            image_string = get_image_string(
                self.tokenizer, splitted_image_counts, self.mp_image_token_length
            )
            messages[0]["content"] = image_string + messages[0]["content"]

        return messages

    def _process_images(
        self, images: List[Image.Image]
    ) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
        """Process and split images for model input.

        :param images: List of PIL images
        :return: Tuple of (processed image tensors, split counts)
        """
        processed_images = []
        splitted_image_counts = []
        for image in images:
            if isinstance(image, Image.Image):
                if image.mode != "RGB":
                    image = image.convert("RGB")
                processed_image, splitted_image_count = self.image_processor(image)
                if (
                    not hasattr(self.tokenizer, "global_image_token")
                    and splitted_image_count[0] * splitted_image_count[1]
                    == len(processed_image) - 1
                ):
                    # If the tokenizer doesn't have a global image token, but the processor generated it, remove it
                    processed_image = processed_image[1:]
                processed_images.append(processed_image)
                splitted_image_counts.append(splitted_image_count)
            else:
                raise ValueError(f"Error processing image: {image}")
        return processed_images, splitted_image_counts

    def _prepare_inputs_and_loss_mask(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare tokenized inputs and create mask for loss computation.

        :param messages: List of conversation messages
        :return: Tuple of (input_ids, loss_mask, attention_mask)
        """
        conv_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=False,
            return_dict=True,
        )
        mask = [0] * len(conv_ids["input_ids"])

        # Locate each assistant turn and flip its mask to 1
        cursor = 0
        for msg in messages:
            segment_ids = self.tokenizer.apply_chat_template(
                [msg], tokenize=True, add_special_tokens=False
            )
            seg_len = len(segment_ids)

            if msg["role"] == "assistant":
                start = cursor + self.prefix_len
                end = cursor + seg_len
                mask[start:end] = [1] * (end - start)  # attend to these tokens

            cursor += seg_len

        return (
            torch.tensor(conv_ids["input_ids"]),
            torch.tensor(mask).to(torch.bool),
            torch.tensor(conv_ids["attention_mask"]),
        )


class VQADataset(BaseDataset):  # Visual Question Answering Dataset
    """Dataset for visual question answering tasks with image-text pairs."""

    def iter_for_worker(self, worker_id: int, num_workers: int) -> Any:
        """Iterate over dataset subset for distributed workers.

        :param worker_id: ID of current worker
        :param num_workers: Total number of workers
        :return: Generator of processed data items
        """
        # dataset = split_dataset_by_node(self.dataset, rank=worker_id, world_size=num_workers)
        for data in self.dataset:
            yield self._process_data(data)

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """Get processed item from dataset.

        :param idx: Index of item to retrieve
        :return: Dictionary with images, input_ids, attention_mask, labels
        """
        item = self.dataset[idx]
        return self._process_data(item)

    def _process_data(self, item: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        """Process single dataset item into model inputs.

        :param item: Raw dataset item
        :return: Processed tensors ready for model input
        """
        # Handle images (should be a list)
        if item["images"] is None:
            images_data = []
        else:
            images_data = item["images"]
            if not isinstance(images_data, list):
                images_data = [images_data]

        processed_images = []
        splitted_image_counts = []
        if images_data:  # Only process if there are images
            processed_images, splitted_image_counts = self._process_images(images_data)

        messages = self._get_messages(item, splitted_image_counts)

        if len(messages) == 0:
            return None

        input_ids, mask, attention_mask = self._prepare_inputs_and_loss_mask(messages)
        labels = self._get_labels(input_ids, mask)

        return {
            "images": processed_images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _get_labels(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Create labels for language modeling loss computation.

        :param input_ids: Token IDs [seq_len]
        :param mask: Boolean mask for loss computation [seq_len]
        :return: Labels tensor with -100 for ignored tokens [seq_len]
        """
        labels = input_ids.clone().masked_fill(~mask, -100)
        labels = labels.roll(-1)  # Shift labels for causal LM
        labels[-1] = -100  # Last token has no target

        return labels
