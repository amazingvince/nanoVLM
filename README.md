# nanoVLM

![nanoVLM](assets/nanoVLM.png)

<a target="_blank" href="https://colab.research.google.com/github/huggingface/nanoVLM/blob/main/nanoVLM.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

---

> [!TIP]
> We have written a [tutorial on nanoVLM](https://huggingface.co/blog/nanovlm) which will guide you through the repository and help you get started in no time.

---

> [!NOTE]
> We have pushed some breaking changes to the repository on June 4. To enable us to do smarter packing, we refactored the way image and text embeddings are combined. To keep everything as smooth as possible, we have trained a new nanoVLM-450M with this new pipeline, while leaving the old nanoVLM-222M compatible with the old pipeline If you clone this repository now or pull the updated to your local machine, the default will be the new 450M Model. If you would like a simpler understanding and a simpler codebase, you can use the v0.1 release. This works out of the box with the old 222M model.

---

nanoVLM is the simplest repository for training/finetuning a small sized Vision-Language Model with a lightweight implementation in pure PyTorch. The code itself is very readable and approachable, the model consists of a Vision Backbone (`models/vision_transformer.py` ~150 lines), Language Decoder (`models/language_model.py` ~250 lines), Modality Projection (`models/modality_projection.py` ~50 lines) and the VLM itself (`models/vision_language_model.py` ~100 lines) and a simple training loop (`train.py` ~200 lines).

Similar to Andrej Karpathy's nanoGPT, we wanted to equip the community with a very simple implementation and training script for Vision Language Models. We do not claim this to be a new SOTA model, rather an educational effort that packs quite a bit of punch if you have the right hardware! You should be able to tweak and play around with the code in no time.

## Supported Model Architectures

nanoVLM now supports flexible combinations of vision encoders and language models:

### Vision Encoders
- **SigLIP** (default): Google's vision transformer optimized for vision-language tasks
  - Model: `google/siglip2-base-patch16-512`
  - Input: 512×512 images → 1024 patches → 64 tokens after projection
  - Position embeddings: Learned
  
- **DINOv3**: Meta's latest self-supervised vision transformer with high-resolution support
  - Models: 
    - `facebook/dinov3-vits16plus-pretrain-lvd1689m` (default)
    - `facebook/dinov3-vitb16-pretrain-lvd1689m` 
    - `facebook/dinov3-vitl14-pretrain-lvd1689m`
  - Input: Aspect-preserving resize up to 1024px → Dynamic patches + 4 registers + 1 CLS → Dynamic tokens
  - Token counts: Scale with image size (e.g., 768×1024 → 768 tokens, 960×960 → 900 tokens)
  - Position embeddings: RoPE with patch center coordinates (enables arbitrary resolutions)
  - Features: LayerScale, SwiGLU FFN, register tokens, rectangular grid support
  
- **DINOv2**: Previous generation DINO model
  - Model: `facebook/dinov2-vits14`
  - Input: Fixed 224×224 images
  - Position embeddings: Learned

### Language Models
- **SmolLM** (default): Hugging Face's efficient small language model
  - Models: `SmolLM2-135M-Instruct`, `SmolLM2-360M-Instruct`
  - Architecture: Llama-style with RMSNorm
  
- **Gemma**: Google's efficient language models
  - Model: `google/gemma-3-270m-it`
  - Architecture: Custom attention with head_dim=256, RoPE
  - Vocabulary: 256k tokens


## What can nanoVLM do?

The model definition and training logic of this repository fits in ~750 lines, with some more boilerplate logging and parameter loading. 
Using the [`SigLIP-B/16-224-85M`](https://huggingface.co/google/siglip-base-patch16-224) and [`HuggingFaceTB/SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) as backbones results in a **222M** nanoVLM. Training this for ~6h on a single H100 GPU on ~1.7M samples of [the cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) results in an accuracy of 35.3% on MMStar.

![loss](assets/nanoVLM-222M-loss.png)

It is therefore a simple yet powerful platform to get started with VLMs. Perfect to tinker around with different setups and settings, to explore the capabilities and efficiencies of small VLMs!

## Quick Start

You can either clone the repository, setup an environment and start with the scripts, or directly [open in Colab](https://colab.research.google.com/github/huggingface/nanoVLM/blob/main/nanoVLM.ipynb). You can also use the [interactive notebook](./nanoVLM.ipynb) to get started!


## Environment Setup

We really like `uv` and recommend using it as your package manager. But feel free to use whichever you prefer.

Let's first clone the repository:
```bash
git clone https://github.com/huggingface/nanoVLM.git
cd nanoVLM
```

If you want to use `uv`:
```bash
uv init --bare --python 3.12
uv sync --python 3.12
source .venv/bin/activate
uv add torch numpy torchvision pillow datasets huggingface-hub transformers wandb einops hf-transfer
# Optional: for lmms-eval integration you have to install it from source, see section 'Evaluation with lmms-eval'
```

If you prefer another environment manager, simply install these packages:  
```bash
pip install torch numpy torchvision pillow datasets huggingface-hub transformers wandb einops hf-transfer
# Optional: for lmms-eval integration you have to install it from source, see section 'Evaluation with lmms-eval'

```
Dependencies: 
- `torch` <3
- `numpy` <3
- `torchvision` for the image processors
- `pillow` for image loading
- `datasets` for the training datasets
- `huggingface-hub` & `transformers` to load the pretrained backbones
- `wandb` for logging

## Training

To train nanoVLM, you can simply use the provided training script. After training, your model gets uploaded to the Hub!

### Default Configuration (SigLIP + SmolLM)
```bash
wandb login --relogin
huggingface-cli login
python train.py
```
This uses the default `models/config.py` with SigLIP vision encoder and SmolLM language model.

### Alternative Configurations

The repository now supports multiple vision encoder and language model combinations:

#### DINOv3 + Gemma-3-270M
```bash
python train.py --use_preset dinov3_gemma

# Or with specific DINOv3 model size
python train.py --vision_encoder dinov3 --dinov3_model vitb16 --language_model gemma
```
This configuration uses:
- **Vision**: DINOv3 with RoPE position embeddings
  - Available sizes: `vits16plus` (default), `vitb16`, `vitl14`
- **Language**: Gemma-3-270M-IT (google/gemma-3-270m-it)
- **Image processing**: High-resolution support up to 1024px with dynamic token grids

#### Custom Vision/Language Combinations
```bash
# Use DINOv3 vision with default SmolLM language model
python train.py --vision_encoder dinov3 --language_model smollm

# Use DINOv2 vision with Gemma language model
python train.py --vision_encoder dinov2 --language_model gemma

# Available options:
# --vision_encoder: siglip (default), dinov3, dinov2
# --language_model: llama/smollm (default), gemma
```

### Training Options

> [!TIP]
> For optimal performance, use `--freeze_vision_encoder` when training with DINOv3, as recommended in the paper.

#### Basic Configuration
```bash
# Adjust learning rates
python train.py --lr_mp 0.00512 --lr_backbones 5e-05

# Set batch size and gradient accumulation
python train.py --batch_size 2 --gradient_accumulation_steps 32

# Enable/disable compilation (disabled by default for compatibility)
python train.py --compile

# Show training progress in console
python train.py --console_log_interval 10  # Log every 10 steps
```

#### Checkpointing and Validation

> [!IMPORTANT]
> `validation_steps` and `save_checkpoint_steps` are independent. Validation computes loss only, while checkpoint saving writes the model to disk.

```bash
# Save checkpoints at regular intervals
python train.py --save_checkpoint_steps 500  # Save every 500 steps

# Run validation at custom intervals (separate from checkpoint saving)
python train.py --validation_steps 100  # Validate every 100 steps

# Control evaluation interval (for best model tracking)
python train.py --eval_interval 500  # Default: 500 steps

# Resume from checkpoint
python train.py --resume_from_vlm_checkpoint --vlm_checkpoint_path checkpoints/run_name
```

#### Advanced Training Strategies
```bash
# Freeze vision encoder (Locked-image Text tuning, recommended for DINOv3)
python train.py --freeze_vision_encoder

# DINOv3 with frozen encoder (as per paper recommendations)
python train.py --use_preset dinov3_gemma --freeze_vision_encoder

# Disable Weights & Biases logging
python train.py --no_log_wandb

# Disable HuggingFace Hub push (no automatic upload after training)
python train.py --hf_repo_name None
```

#### Performance Tuning
```bash
# Control number of DataLoader workers
python train.py --num_workers 4  # Default: 2

# Limit maximum threads for torch operations
python train.py --max_threads 8  # Default: 4
```

## Generate

To try a [trained model](https://huggingface.co/lusxvr/nanoVLM-450M), you can simply use the provided generate script
```bash
python generate.py
```
or, to use your own trained model, you can simply run:
```bash
python generate.py --checkpoint /your/path/to/trained_models
```

### Validating Checkpoints

You can validate and test generation from saved checkpoints:
```bash
# Validate a checkpoint and generate sample outputs
python validate_checkpoint.py --checkpoint_dir checkpoints/your_run_name

# Test different architectures
python validate_checkpoint.py --checkpoint_dir checkpoints/dinov3_run --vision_encoder dinov3
```

If we feed the example image in `assets/image.png` with a question into the model, we get the following output. Even after only short training, the model can recognize the cat in the picture. 
```
Input: 
Image + 'What is this?'

Outputs:
Generation 1:  This is a cat sitting on the ground. I think this is a cat sitting on the ground.
Generation 2:  This picture is clicked outside. In the center there is a brown color cat seems to be sitting on
Generation 3:  This is a cat sitting on the ground, which is of white and brown in color. This cat
Generation 4:  This is a cat sitting on the ground. I think this is a cat sitting on the ground.
Generation 5:  This is a cat sitting on the ground, which is covered with a mat. I think this is
```

## Using Trained Models for Inference

After training, you can use your model with the standard transformers library:

```python
from models.vision_language_model import VisionLanguageModel
from models import config
from data.processors import get_tokenizer, get_image_processor
from PIL import Image
import torch

# Load checkpoint
checkpoint_dir = "checkpoints/your_run_name"
model = VisionLanguageModel.from_pretrained(checkpoint_dir)
model.eval()

# Setup tokenizer and processor (recreated from config)
cfg = model.cfg
tokenizer = get_tokenizer(cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template)
image_processor = get_image_processor(cfg)

# Process image and text
image = Image.open("path/to/image.jpg")
processed_image = image_processor(image)
prompt = "What is in this image?"

# Generate response
response = model.generate(
    processed_image,
    prompt,
    tokenizer,
    max_new_tokens=100,
    temperature=0.7
)
print(response)
```

For a complete example, see `inference_example.py` in the repository.

### Evaluation with lmms-eval

nanoVLM supports evaluation using the comprehensive [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) toolkit:

```bash
# Install lmms-eval (has to be from source)
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git

# Make sure you have your environment variables set correctly and you are logged in to HF
export HF_HOME="<Path to HF cache>"
huggingface-cli login

# Evaluate a trained model on multiple benchmarks
python evaluation.py --model lusxvr/nanoVLM-450M --tasks mmstar,mme

> [!NOTE]
> lmms-eval is disabled by default during training as it's computationally expensive. Enable it explicitly with `--use_lmms_eval` when you want benchmark scores.

```bash
# Enable lmms-eval during training
python train.py --use_lmms_eval --lmms_eval_tasks "mmstar,mme" --lmms_eval_limit 1000
```

# If you want to use it programmatically:
from evaluation import cli_evaluate
import argparse

args = argparse.Namespace(
    model='lusxvr/nanoVLM-450M', # This can be either a checkpoint path or the model itself
    tasks='mmstar,mmmu,ocrbench',
    batch_size=128 # Adapt this to your GPU, needs to be passed to avoid an OOM Error
)
results = cli_evaluate(args)
```

## Hub integration

**nanoVLM** comes with handy methods to load and save the model from the Hugging Face Hub.

### Pretrained weights

Here is how to load from a repo on the Hugging Face Hub. This is the recommended way to start working with the pretrained weights.

```python
# Load pretrained weights from Hub
from models.vision_language_model import VisionLanguageModel

model = VisionLanguageModel.from_pretrained("lusxvr/nanoVLM-450M")
```

### Push to hub

Once you've trained a **nanoVLM** model, you might want to share it on the Hugging Face Hub. You can easily do that with:

```python
... # Load and train your model

# Push it to `username/my-awesome-nanovlm-model` repo
model.push_to_hub("my-awesome-nanovlm-model")
```

The model will be saved on the Hub as a config file `config.json` and a weights file `model.safetensors`. A modelcard `README.md` will also be generated for you with some high-level information. Feel free to update it manually to explain your work.

If the repo does not exist, it will be created for you. By default the repo will be public. You can pass `private=True` if you don't want to share publicly.


### Local save/load

If you don't want to host your model on the Hugging Face Hub, it is still possible to save it locally:

```python
... # Load and train your model

# Save it to a local folder
model.save_pretrained("path/to/local/model")
```

You can then reload it from the local path:

```python
# Load pretrained weights from local path
from models.vision_language_model import VisionLanguageModel

model = VisionLanguageModel.from_pretrained("path/to/local/model")
```

## VRAM Usage

Understanding the VRAM requirements for training is crucial for selecting the right hardware and batch sizes. We've benchmarked the default `nanoVLM` model (222M parameters) on a single NVIDIA H100 GPU. Below is a summary of the peak VRAM usage observed for different batch sizes during training (including model, gradients, and optimizer states):

<img src="assets/VRAM_Usage_vs_Batch_Size_nanoVLM.png" width="600" alt="VRAM Usage vs Batch Size">

Here's a breakdown of the approximate peak VRAM usage:

```
VRAM allocated after loading model to device: 871.44 MB
--- Summary of VRAM Usage ---
Batch Size 1: 4448.58 MB
Batch Size 2: 4465.39 MB
Batch Size 4: 4532.29 MB
Batch Size 8: 5373.46 MB
Batch Size 16: 7604.36 MB
Batch Size 32: 12074.31 MB
Batch Size 64: 20995.06 MB
Batch Size 128: 38834.19 MB
Batch Size 256: 74561.08 MB
Batch Size 512: OOM (Peak before OOM: 80247.67 MB)
```

Note that the VRAM measurement was performed on a small setup using 'SmolLM2-135M' with a maximum input sequence length of 128 tokens. This may differ from the current default configuration in the project.

> [!WARNING]
> You'll need at least ~4.5 GB of VRAM to train the default model even with a batch size of 1. With 8 GB of VRAM, you can train with batch sizes up to 16.

**Measure for Your Setup:**

The values above are for the default model configuration. If you modify the model architecture (e.g., change backbones, hidden sizes) or use different sequence lengths, your VRAM requirements will change. 

We provide a script `measure_vram.py` that allows you to test VRAM requirements on your specific machine and for your chosen model configuration and batch sizes. 

To use it:
1. Ensure you have a CUDA-enabled GPU and PyTorch installed.
2. Run the script with your desired batch sizes. You can also specify a model checkpoint if you have one, or let it initialize a new model based on the default `VLMConfig`.

```bash
# Example: Test batch sizes 1, 2, 4, 8 with a new default model
python measure_vram.py --batch_sizes "1 2 4 8"

# Example: Test with a specific checkpoint and different batch sizes
python measure_vram.py --vlm_checkpoint_path path/to/your/model.pth --batch_sizes "16 32 64"

```

This script will output the peak VRAM allocated for each batch size tested, helping you determine feasible training configurations for your hardware.


## Contributing

We welcome contributions to nanoVLM! However, to maintain the repository's focus on simplicity and pure PyTorch, we have a few guidelines:

> [!IMPORTANT]
> nanoVLM is intentionally kept simple and in pure PyTorch. Contributions that introduce dependencies like `transformers.Trainer`, `accelerate`, or `deepspeed` will not be accepted.

*   **Pure PyTorch:** We aim to keep nanoVLM as a lightweight implementation in pure PyTorch.
*   **New Features:** If you have an idea for a new feature, please open an issue first to discuss the scope and implementation details. This helps ensure that your contribution aligns with the project's goals.
*   **Bug Fixes:** Feel free to submit pull requests for bug fixes.

### Roadmap

Here are some areas we're looking to work on in the near future. Contributions in these areas are particularly welcome:

*   **Evaluations:** Implementing more evaluations or improving our MMStar implementation (highly valued)
*   **Data Packing:** Implementing a way to create packs of a given size from the input data to optimize training.
*   **Multi-gpu training:** Training on several GPUs
*   **Multi-image support:** Training with several images
*   **Image-splitting:** Enabling higher resolutions through image-splitting as done in SmolVLM.
*   **VLMEvalKit:** Integration into [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) to enable further benchmarks

## Citation

If you like the project and want to use it somewhere, please use this citation:
```
@misc{wiedmann2025nanovlm,
  author = {Luis Wiedmann and Aritra Roy Gosthipaty and Andrés Marafioti},
  title = {nanoVLM},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/nanoVLM}}
}
```
