import torch
from PIL import Image
from torch.utils.data import Dataset

from data.processors import get_image_string


class BaseDataset(Dataset):
    def __init__(self, dataset, tokenizer, image_processor, mp_image_token_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length
        # Add config and grid factory
        self.cfg = getattr(dataset, 'cfg', None)
        if hasattr(dataset, 'grid_factory') and dataset.grid_factory is not None:
            self.grid_factory = dataset.grid_factory
        else:
            from models.grid_abstraction import GridFactory
            self.grid_factory = GridFactory()
            # Ensure cfg is set for grid factory
            if self.cfg is None and hasattr(dataset, 'cfg'):
                self.cfg = dataset.cfg
        
        self.prefix_len = self._get_prefix_len()

    def __len__(self):
        return len(self.dataset)

    def _get_prefix_len(self):
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

    def _get_messages(self, item, splitted_image_counts):
        messages = []
        for text in item["texts"]:
            messages.append({"role": "user", "content": text["user"]})
            messages.append({"role": "assistant", "content": text["assistant"]})

        image_string = get_image_string(
            self.tokenizer, splitted_image_counts, self.mp_image_token_length, getattr(self, 'cfg', None)
        )

        if len(splitted_image_counts) > 0:
            messages[0]["content"] = image_string + messages[0]["content"]

        return messages

    def _process_images(self, images):
        processed_images = []
        splitted_image_counts = []
        for image in images:
            if isinstance(image, Image.Image):
                if image.mode != "RGB":
                    image = image.convert("RGB")
                processed_image, splitted_image_count = self.image_processor(image)
                processed_images.append(processed_image)
                splitted_image_counts.append(splitted_image_count)
            else:
                raise ValueError("Error processing image")
        return processed_images, splitted_image_counts

    def _prepare_inputs_and_loss_mask(self, messages):
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
    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Handle images (should be a list)
        images_data = item["images"]
        if not isinstance(images_data, list):
            images_data = [images_data]

        # Now process the images
        processed_images, splitted_image_counts = self._process_images(images_data)

        # Extract GhGw from the grid dict for DINOv3
        if len(splitted_image_counts) > 0 and isinstance(
            splitted_image_counts[0], dict
        ):
            splitted_image_counts = [grid["GhGw"] for grid in splitted_image_counts]

        messages = self._get_messages(item, splitted_image_counts)

        input_ids, mask, attention_mask = self._prepare_inputs_and_loss_mask(messages)
        labels = self._get_labels(input_ids, mask)

        return {
            "images": processed_images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "image_grids": splitted_image_counts,  # Pass grid dimensions
        }

    def _get_labels(self, input_ids, mask):
        # Ensure mask length matches input_ids length
        if len(mask) != len(input_ids):
            # Pad or truncate mask to match input_ids length
            if len(mask) < len(input_ids):
                mask = torch.cat(
                    [
                        mask,
                        torch.zeros(
                            len(input_ids) - len(mask),
                            dtype=torch.bool,
                            device=mask.device,
                        ),
                    ]
                )
            else:
                mask = mask[: len(input_ids)]

        labels = input_ids.clone().masked_fill(~mask, -100)
        labels = labels.roll(-1)  # Shift labels for causal LM
        if len(labels) > 0:
            labels[-1] = -100  # Last token has no target

        return labels
