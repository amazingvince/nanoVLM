from data.datasets import VQADataset
from data.processors import get_image_processor, get_tokenizer
from models.config import VLMConfig

cfg = VLMConfig(vit_architecture='dinov3')
tokenizer = get_tokenizer(cfg.lm_tokenizer, cfg.vlm_extra_tokens)
image_processor = get_image_processor(cfg.max_img_size, cfg.vit_img_size, single_image_mode=True, vit_patch_size=cfg.vit_patch_size, pixel_shuffle_factor=cfg.mp_pixel_shuffle_factor)
from datasets import load_dataset

ds = load_dataset('HuggingFaceM4/the_cauldron', 'vqav2', split='train[:10]')
dataset = VQADataset(ds, tokenizer, image_processor, cfg.mp_image_token_length)
print(f'Initial dataset size: {len(dataset)}')
max_len = cfg.lm_max_position_embeddings
filtered_indices = [i for i in range(len(dataset)) if len(dataset[i]['input_ids']) <= max_len]
print(f'Filtered dataset size: {len(filtered_indices)}')
print(f'Filtered out: {len(dataset) - len(filtered_indices)} samples')
for i in range(min(5, len(dataset))):
    item = dataset[i]
    print(f'Sample {i} length: {len(item['input_ids'])}, exceeds max: {len(item['input_ids']) > max_len}')
