import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn


# Used to check our models performance on multiple choice tasks. This can also be done in a more involved way with e.g. LLM-as-a-judge
def check_multiple_choice_with_regex(model_outputs, correct_answers):
    results = []
    for model_output, correct_answer in zip(model_outputs, correct_answers):
        # Strip any trailing newlines and convert to uppercase
        correct_answer = correct_answer.rstrip("\n").upper()

        # Look for the answer letter at the beginning of a line or as the last word
        patterns = [
            rf"\b{correct_answer}\b",  # Word boundary around the answer letter
            rf"\b{correct_answer}[.,)]",  # Answer followed by punctuation
            rf"\(.*{correct_answer}.*\)",  # Answer within parentheses
        ]

        match_found = False
        for pattern in patterns:
            if re.search(pattern, model_output):
                match_found = True
                break  # Exit inner loop once a match is found
        results.append(match_found)
    return results


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf")):
    """
    Apply top-k and/or nucleus (top-p) filtering to logits.
    """
    top_k = min(top_k, logits.size(-1))  # Safety

    if top_k > 0:
        # Remove all tokens with a probability less than the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative probability above top_p
        sorted_indices_to_remove = cumulative_probs > top_p

        # Always keep the first token
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits


def configure_tf32() -> bool:
    """
    Enable TF32 precision for GPUs with compute capability >= 8.0 (Ampere+).
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


@dataclass
class _LayerSummary:
    name: str
    param_shape: Optional[torch.Size]
    inclusive_total_params: int
    inclusive_trainable_params: int


def model_summary(
    model: nn.Module, max_depth: int = 4, show_param_shapes: bool = False
) -> None:
    """
    Prints a hierarchical summary of a PyTorch model with *inclusive* parameter counts.
    Counts are robust to shared/tied parameters (each Parameter is counted once per subtree).
    """

    # ---------- formatting helpers ----------
    def _format_number(num: int) -> str:
        return f"{num:,}" if num > 0 else "--"

    def _format_shape(shape: Optional[torch.Size]) -> str:
        return "x".join(map(str, shape)) if shape else "N/A"

    # ---------- build param info once ----------
    # Map: id(param) -> (numel, requires_grad)
    param_info: Dict[int, Tuple[int, bool]] = {}
    for p in model.parameters(recurse=True):
        pid = id(p)
        if pid not in param_info:
            param_info[pid] = (p.numel(), bool(p.requires_grad))

    # Fast path: totals only
    if max_depth <= 0:
        total_params = sum(n for (n, _) in param_info.values())
        trainable_params = sum(n for (n, rg) in param_info.values() if rg)
        print("=" * 50)
        print("Total params:", _format_number(total_params))
        print("Trainable params:", _format_number(trainable_params))
        nontrain = total_params - trainable_params
        print("Non-trainable params:", _format_number(nontrain))
        print("=" * 50)
        return

    summary_list: List[_LayerSummary] = []

    def summarize_recursive(module: nn.Module, depth: int, prefix: str) -> Set[int]:
        """
        Return the set of unique Parameter IDs reachable from this module's subtree.
        Also appends a _LayerSummary for this module.
        """
        # If we're beyond the print depth, just return the deduped set upward
        if depth > max_depth:
            ids = {id(p) for p in module.parameters(recurse=True)}
            return ids

        # Direct parameters of *this* module (non-recursive)
        direct_ids: Set[int] = {id(p) for p in module.parameters(recurse=False)}

        # Recurse into children and union their sets
        child_ids: Set[int] = set()
        for child in module.children():
            child_ids |= summarize_recursive(child, depth + 1, prefix + "  ")

        all_ids = direct_ids | child_ids

        # Inclusive counts from the deduped set
        total = sum(param_info[i][0] for i in all_ids)
        trainable = sum(param_info[i][0] for i in all_ids if param_info[i][1])

        # First direct trainable parameter shape (display purpose only)
        param_shape = next(
            (p.shape for p in module.parameters(recurse=False) if p.requires_grad),
            None,
        )

        summary_list.append(
            _LayerSummary(
                name=f"{prefix}{type(module).__name__}",
                param_shape=param_shape,
                inclusive_total_params=total,
                inclusive_trainable_params=trainable,
            )
        )
        return all_ids

    # Build the list (pre-order traversal)
    summarize_recursive(model, 1, "")

    # Totals from the whole model (already deduped)
    total_params = sum(n for (n, _) in param_info.values())
    trainable_params = sum(n for (n, rg) in param_info.values() if rg)

    # ---------- printing ----------
    name_col_width = max(len("Layer (type)"), max(len(s.name) for s in summary_list))
    shape_col_width = 0
    if show_param_shapes:
        shape_col_width = max(
            len("Param Shape"),
            max(len(_format_shape(s.param_shape)) for s in summary_list),
        )

    params_col_width = 12
    trainable_col_width = 10
    col_spacing = "  "

    header_parts = [f"{'Layer (type)':<{name_col_width}}"]
    if show_param_shapes:
        header_parts.append(f"{'Param Shape':>{shape_col_width}}")
    header_parts.append(f"{'Param #':>{params_col_width}}")
    header_parts.append(f"{'Trainable':>{trainable_col_width}}")
    header = col_spacing.join(header_parts)
    sep = "=" * len(header)

    print(sep)
    print(header)
    print(sep)
    for e in summary_list:
        parts = [f"{e.name:<{name_col_width}}"]
        if show_param_shapes:
            parts.append(f"{_format_shape(e.param_shape):>{shape_col_width}}")
        parts.append(f"{_format_number(e.inclusive_total_params):>{params_col_width}}")
        parts.append(f"{str(e.inclusive_trainable_params > 0):>{trainable_col_width}}")
        print(col_spacing.join(parts))
    print(sep)
    print(f"Total params: {_format_number(total_params)}")
    print(f"Trainable params: {_format_number(trainable_params)}")
    print(f"Non-trainable params: {_format_number(total_params - trainable_params)}")
    print(sep)
