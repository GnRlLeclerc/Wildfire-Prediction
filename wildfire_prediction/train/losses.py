from torch import Tensor
import torch.nn.functional as F


def mse_scaled_loss(
    input_logits: Tensor, target_logits: Tensor, temperature: float = 0.5
):
    """Computes mse loss between with temperature scaling"""

    # Apply temperature scaling
    input_probs = F.softmax(input_logits / temperature, dim=-1)
    target_probs = F.softmax(target_logits / temperature, dim=-1)

    # MSE
    return F.mse_loss(input_probs, target_probs, reduction="mean")


def kl_divergence_loss(
    input_logits: Tensor, target_logits: Tensor, temperature: float = 0.5
):
    """Computes Kullback-Leibler (KL) divergence loss with temperature scaling"""

    # Apply temperature scaling
    target_logits_scaled = target_logits / temperature

    # Get probability distributions
    input_probs = F.log_softmax(input_logits, dim=-1)
    output_probs = F.softmax(target_logits_scaled, dim=-1)

    # KL divergence
    return F.kl_div(input_probs, output_probs, reduction="batchmean")
