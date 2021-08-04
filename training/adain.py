import torch.nn as nn
import torch
import torch.nn.functional as F


def adain(source, target):
    batch_size, channels, width, height = _check_shapes(source, target)
    source_normalized = F.instance_norm(source)
    target_stdevs, target_means = calc_feature_stats(target)
    source_stats_matched = _match_normalized_to_stats(source_normalized, target_stdevs, target_means)

    assert source_stats_matched.shape == (batch_size, channels, width, height)
    return source_stats_matched


def calc_feature_stats(target):
    batch_size, channels, w, h = target.shape
    target_reshaped = target.view(batch_size, channels, 1, 1, -1)
    target_stdevs = target_reshaped.var(-1, unbiased = False) ** 0.5
    target_means = target_reshaped.mean(-1)

    assert_shape(target_stdevs, (batch_size, channels, 1, 1))
    assert_shape(target_means, (batch_size, channels, 1, 1))

    return target_stdevs, target_means


def _match_normalized_to_stats(normalized, target_stdevs, target_means):
    return normalized * target_stdevs + target_means


def _check_shapes(source, target):
    assert len(target.shape) == 4, "expected 4 dimensions"
    assert target.shape[:2] == source.shape[:2], \
            "source/target batch or channels mismatch"
    batch_size, channels, width, height = source.shape

    return batch_size, channels, width, height


def assert_shape(tensor, expected):
    msg = f"expected {expected} actual {tensor.shape}"
    assert tensor.shape == expected, msg
