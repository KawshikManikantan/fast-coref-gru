import torch
from typing import Tuple
from torch import Tensor


def sort_mentions(
    ment_starts: Tensor, ment_ends: Tensor, return_sorted_indices=False
) -> Tuple:
    # print("Entered Sort Mentions #############################")
    # print(ment_starts)
    # print(ment_ends)
    sort_scores = ment_starts.to(torch.float64) + 1e-5 * ment_ends.to(torch.float64)
    # print(ment_starts.dtype)
    # print(sort_scores)
    _, sorted_indices = torch.sort(sort_scores, 0)
    # print(sorted_indices)
    output: Tuple = (ment_starts[sorted_indices], ment_ends[sorted_indices])
    # print(ment_starts[sorted_indices])
    # print(ment_ends[sorted_indices])
    if return_sorted_indices:
        output = output + (sorted_indices,)

    return output
