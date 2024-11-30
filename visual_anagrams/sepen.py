import torch

def anti_seg_loss(amaps: torch.Tensor) -> torch.Tensor:
    '''
    Computes the anti-separation loss between attention maps
    amaps (torch.Tensor) :
        Tensor of shape (T, H, W) representing the attention maps
    '''
    t, h, w = amaps.shape
    amaps = amaps.view(t, -1)
    loss = 0
    for i in range(t):
        for j in range(i + 1, t):
            loss += torch.abs(0.5 - (t - 1) * 0.05 - torch.min(amaps[i], amaps[j]).sum() / torch.sum(amaps[i] + amaps[j]))
    loss /= t * (t - 1) / 2
    return loss