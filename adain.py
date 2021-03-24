import torch.nn as nn
import torch
import torch.nn.functional as F



def main():
    target = torch.randint(-20, 20, (8, 3, 4, 4)).float()
    source = torch.randint(-20, 20, (8, 3, 4, 4)).float()

    stylized_source = adain(source, target)

    target = target.view(8, 3, -1)
    stylized_source = stylized_source.view(8, 3, -1)

    # check variances the same
    target_variances = target.var(-1)
    stylized_variances = target.var(-1)
    assert torch.all(torch.abs(stylized_variances - target_variances) < 1e-6)

    # check means the same
    target_means = target.mean(-1)
    stylized_means = target.mean(-1)
    assert torch.all(torch.abs(stylized_means - target_means) < 1e-6)

    # check values are just rescaled / shifted
    diff = F.instance_norm(source) - F.instance_norm(stylized_source.view(8, 3, 4, 4))
    assert torch.all(torch.abs(diff) < 1e-6)


def adain(source, target):
    # check shapes
    assert len(target.shape) == 4, "expected 4 dimensions"
    assert target.shape == source.shape, "source/target shape mismatch"
    batch_size, channels, width, height = source.shape

    # calculate target stats
    #target = target.view(batch_size, channels, 1, 1, -1) #dont work for export
    target_reshaped = target.view(1, channels, 1, 1, -1)
    target_variances = target_reshaped.var(-1)
    target_means = target_reshaped.mean(-1)

    # normalize and rescale source to match target stats
    normalized = F.instance_norm(source)
    result = source * (target_variances ** 0.5) + target_means

    assert result.shape == (batch_size, channels, width, height)
    return result



if __name__ == "__main__":
    main()
