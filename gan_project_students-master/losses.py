from typing import Tuple

import torch
import torch.nn as nn

from options import Options


def calc_vae_loss(distance: nn.Module, input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                  target, options: Options) -> torch.Tensor:
    """
    calculates the vae loss
    :param options: Options
    :param distance: The distance to use for reconstruction loss
    :param input: tuple consisting of (decoded image, latent_vector, mu, log_var)
    :param target: target image (torch.Tensor)
    :rtype vae_loss: torch.Tensor
    """
    pass
