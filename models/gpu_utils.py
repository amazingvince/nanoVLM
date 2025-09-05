import logging

import torch


def configure_tf32():
    """
    Enable TF32 precision for GPUs with compute capability >= 8.0 (Ampere+).

    Returns:
        bool: True if TF32 was enabled, False otherwise
    """
    if not torch.cuda.is_available():
        logging.info("No GPU detected, running on CPU.")
        return False

    try:
        device = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device)
        major, minor = capability
        gpu_name = torch.cuda.get_device_name(device)

        if major >= 8:
            # Modern API - replaces both backend flags
            torch.set_float32_matmul_precision("high")
            logging.info(f"{gpu_name} (compute {major}.{minor}) - TF32 enabled")
            return True
        else:
            logging.info(f"{gpu_name} (compute {major}.{minor}) - TF32 not supported")
            return False

    except Exception as e:
        logging.error(f"Failed to configure GPU: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    configure_tf32()
