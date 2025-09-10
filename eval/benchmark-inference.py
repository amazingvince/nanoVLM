import torch
from PIL import Image
from torch.utils import benchmark

from data.processors import get_image_processor, get_tokenizer
from models.vision_language_model import VisionLanguageModel

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def generate_tokens(tokens, image, image_grids=None):
    return model.generate(tokens, image, image_grids=image_grids, max_new_tokens=1000)


if __name__ == "__main__":
    model = VisionLanguageModel.from_pretrained("lusxvr/nanoVLM-450M").to(device)
    model.eval()

    tokenizer = get_tokenizer(model.cfg.lm_tokenizer, model.cfg.vlm_extra_tokens)
    single_image_mode = model.cfg.vit_architecture == "dinov3"
    image_processor = get_image_processor(
        model.cfg.max_img_size,
        model.cfg.vit_img_size,
        single_image_mode=single_image_mode,
        vit_patch_size=model.cfg.vit_patch_size,
        pixel_shuffle_factor=model.cfg.mp_pixel_shuffle_factor,
    )

    text = "What is this?"
    template = f"{tokenizer.image_token * model.cfg.mp_image_token_length}Question: {text} Answer:"
    encoded_batch = tokenizer.batch_encode_plus([template], return_tensors="pt")
    tokens = encoded_batch["input_ids"].to(device)

    image_path = "assets/image.png"
    image = Image.open(image_path)
    processed = image_processor(image)
    # Handle tuple return from processor (image, grid)
    if isinstance(processed, tuple):
        image, grid = processed
    else:
        image = processed
    image = image.unsqueeze(0).to(device)

    # Pass image_grids if available
    timer_globals = {"tokens": tokens, "image": image}
    stmt = "generate_tokens(tokens, image"
    if "grid" in locals():
        timer_globals["image_grids"] = [grid]
        stmt += ", image_grids=image_grids"
    stmt += ")"

    time = benchmark.Timer(
        stmt=stmt,
        setup="from __main__ import generate_tokens",
        globals=timer_globals,
        num_threads=torch.get_num_threads(),
    )

    print(time.timeit(10))
