import torch.nn as nn
import torch
import torch.nn.functional as F



def main():
    batch_size = 8
    channels = 5
    w = 6
    h = 6
    source = torch.randint(-5, 5, (batch_size, channels, w, h)).float()
    target = torch.randint(-5, 5, (batch_size, channels, w, h)).float()

    stylized_source = adain(source, target)

    target = target.view(batch_size, channels, -1)
    stylized_source = stylized_source.view(batch_size, channels, -1)

    # check variances the same
    target_variances = target.var(-1, unbiased = False)
    stylized_variances = stylized_source.var(-1, unbiased = False)
    assert torch.all(torch.abs(stylized_variances - target_variances) < 1e-4), \
            (target_variances, stylized_variances)
    print("Vars match")

    # check means the same
    target_means = target.mean(-1)
    stylized_means = stylized_source.mean(-1)
    assert torch.all(torch.abs(stylized_means - target_means) < 1e-4)
    print("Means match")

    # check values are just rescaled / shifted
    stylized_reshaped = stylized_source.view(batch_size, channels, w, h)
    diff = F.instance_norm(source) - F.instance_norm(stylized_reshaped)
    assert torch.all(torch.abs(diff) < 1e-4)
    print("Vals match")


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



if __name__ == "__main__":
    main()
