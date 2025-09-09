# Using Trained nanoVLM Models

## Loading a Checkpoint

After training, your model checkpoint will be saved in the `checkpoints/` directory with the following structure:

```
checkpoints/
└── nanoVLM_[model_details]/
    └── step_[N]/
        ├── config.json           # Model configuration
        ├── model.safetensors     # Model weights
        ├── tokenizer/            # Tokenizer files
        └── processor/            # Processor configuration
            └── preprocessor_config.json
```

## Basic Inference

### Using the Provided Script

```bash
python inference_example.py checkpoints/your_model/step_1000 path/to/image.jpg --prompt "What's in this image?"
```

### Python API

```python
from inference_example import load_nanovlm_model, generate_response

# Load model
model, tokenizer, image_processor, device = load_nanovlm_model("checkpoints/your_model/step_1000")

# Generate response
response = generate_response(
    model, tokenizer, image_processor,
    "path/to/image.jpg", 
    "Describe this image",
    device
)
print(response)
```

## Integration with Transformers

The saved checkpoints are designed to work with the custom nanoVLM architecture. While not directly compatible with `transformers.AutoModel`, the checkpoint includes:

1. **Model weights** in safetensors format
2. **Tokenizer** in HuggingFace format (fully compatible)
3. **Processor config** that specifies how to recreate the image processor

### Loading Components Separately

```python
from transformers import AutoTokenizer
from models.vision_language_model import VisionLanguageModel
import json

# Load model
model = VisionLanguageModel.from_pretrained("checkpoints/your_model/step_1000")

# Load tokenizer (HuggingFace compatible)
tokenizer = AutoTokenizer.from_pretrained("checkpoints/your_model/step_1000/tokenizer")

# Recreate processor from config
with open("checkpoints/your_model/step_1000/processor/preprocessor_config.json") as f:
    proc_config = json.load(f)
# Use proc_config to initialize the appropriate processor
```

## Processor Details

The image processor configuration varies by architecture:

### SigLIP Models
- Splits large images into multiple patches based on vit_img_size
- Uses dynamic resizing up to max_img_size
- Config saved in `preprocessor_config.json`

### DINOv3 Models  
- Aspect-preserving resize up to max_img_size
- Single image mode with dynamic token grids based on actual image dimensions
- Uses RoPE positional encoding for spatial awareness at any resolution
- For DINOv3, we also attempt to save the HuggingFace processor when available

## Converting for Deployment

For production deployment, you may want to:

1. **Export to ONNX**: Use the model's forward pass for tracing
2. **Quantize**: Apply int8/int4 quantization for efficiency
3. **Create HuggingFace Hub model**: Package with a model card and proper configs

## Validation

To validate a checkpoint's performance:

```bash
python validate_checkpoint.py checkpoints/your_model/step_1000 --batch_size 8
```

This will:
- Run validation on the same dataset split used during training
- Generate sample outputs
- Save metrics to `validation_results.json`