from typing import Dict, List
from allennlp.data.tokenizers import Tokenizer
from allennlp.training.metrics.metric import Metric
from sacrebleu import BLEU
from argparse import Namespace

import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Metric.register("sacre_bleu")
class SacreBleu(Metric):
    """
    TensorBLEU implementation based on sacrebleu for more reproducibility.
    """

    def __init__(
        self,
        detokenizer: Tokenizer,
        force: bool = False,
        lowercase: bool = False,
        tokenize: str = "13a",
        use_effective_order: bool = False,
        get_bp: bool = False,
        smooth_method: str = "exp",
    ):

        self.bleu = BLEU(
            Namespace(force=force, lc=lowercase, smooth_value=None, smooth_method=smooth_method, tokenize=tokenize)
        )

        self.detokenizer = detokenizer
        self.use_effective_order = use_effective_order
        self.get_bp = get_bp

        self._hyp_sentences: List[str] = []
        self._ref_sentences: List[str] = []

    @property
    def signature(self):
        self.bleu.signature.info["numrefs"] = len(self._ref_sentences)
        return self.bleu.signature

    def __call__(self, hypothesis_tokens: List[List[str]], reference_tokens: List[List[str]]):
        self._hyp_sentences += [self.detokenizer.detokenize(tokens) for tokens in hypothesis_tokens]
        self._ref_sentences += [self.detokenizer.detokenize(tokens) for tokens in reference_tokens]

    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        bleu = self.bleu.corpus_score(
            sys_stream=self._hyp_sentences,
            ref_streams=[self._ref_sentences],
            use_effective_order=self.use_effective_order,
        )
        if self.get_bp:
            score = bleu.bp
        else:
            score = bleu.score
        if reset:
            logger.info(f"SacreBLEU Version String: {self.signature}")
            self.reset()
        return score

    def reset(self) -> None:
        self._hyp_sentences = []
        self._ref_sentences = []
