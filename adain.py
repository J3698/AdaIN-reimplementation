import torch.nn as nn
import torch
import torch.nn.functional as F


def adain(source, target):
    # check shapes
    assert len(target.shape) == 4, "expected 4 dimensions"
    assert target.shape == source.shape, "source/target shape mismatch"
    batch_size, channels, width, height = source.shape

    # calculate target stats
    #target = target.view(batch_size, channels, 1, 1, -1) #dont work for export
    target_reshaped = target.view(batch_size, channels, 1, 1, -1)
    target_variances = target_reshaped.var(-1, unbiased = False)
    target_means = target_reshaped.mean(-1)

    # normalize and rescale source to match target stats
    normalized = F.instance_norm(source)

    result = normalized * (target_variances ** 0.5) + target_means

    assert result.shape == (batch_size, channels, width, height)
    return result
