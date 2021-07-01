import math
from overrides import overrides

from allennlp.training.metrics.metric import Metric


class LossToPerplexity(Metric):
    """
    Compute perplexity from loss value.
    """

    def __init__(self):
        self.count = 0
        self.total_loss = 0

    def __call__(self,
                 loss: float,
                 count: int = 1):
        self.total_loss += float(loss)
        self.count += int(count)

    def get_metric(self, reset: bool = False):
        if self.count > 0:
            try:
                score = math.exp(self.total_loss / self.count)
            except OverflowError:
                score = math.inf
        else:
            score = 0

        if reset:
            self.reset()
        return score

    @overrides
    def reset(self):
        self.count = 0
        self.total_loss = 0
