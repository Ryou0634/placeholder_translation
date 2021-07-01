import torch
import torch.nn.functional as F


def sampling_from_logits(logits: torch.Tensor,
                         temperature: float = 1.0,
                         top_k: int = None,
                         top_p: float = None):
    """

    Parameters
    ----------
    logits : torch.Tensor (batch_size, num_classes)
    temperature : int
        The higher the temperature, the distribution gets similar to uniform,
        and the lower the temperature, the distribution gets similar to one-hot.
    top_k : int
        Sample from top_k classes.
    top_p : int
        Sample from top_p classes whose cumulative probability is top_p.


    Returns
    -------
    sampled_class : torch.Tensor (batch_size, )
    """

    batch_size = logits.size(0)

    logits /= temperature

    if top_k:
        top_k_logits, top_k_indices = logits.topk(top_k, dim=1)
        top_k_sampled_class = torch.distributions.Categorical(logits=top_k_logits).sample()
        sampled_class = top_k_indices[torch.arange(batch_size), top_k_sampled_class]

    elif top_p:
        probs = F.softmax(logits, dim=1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=1)
        _cumsum = sorted_probs.cumsum(1)
        mask = _cumsum < top_p
        mask = torch.cat([torch.ones_like(mask[:, :1]), mask[:, :-1]], 1)
        sorted_probs = sorted_probs * mask.float()
        sorted_probs = sorted_probs / sorted_probs.sum(1, keepdim=True)
        logits.scatter_(1, sorted_indices, sorted_probs.log())
        sampled_class = torch.distributions.Categorical(logits=logits).sample()

    else:
        sampled_class = torch.distributions.Categorical(logits=logits).sample()

    return sampled_class
