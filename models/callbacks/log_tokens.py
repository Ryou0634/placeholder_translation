from typing import Dict

from allennlp.data import Vocabulary
from allennlp.training.trainer import TrainerCallback, GradientDescentTrainer
from models.utils import tensor2tokens
from allennlp.nn.util import move_to_device

import logging

logger = logging.getLogger(__name__)


@TrainerCallback.register("log_tokens")
class LogTokens(TrainerCallback):
    def __init__(self, input_name_spaces: Dict[str, str], output_name_spaces: Dict[str, str] = None):
        self.input_name_spaces = input_name_spaces
        self.output_name_spaces = output_name_spaces

    def on_start(self, trainer: "GradientDescentTrainer", is_primary: bool = True, **kwargs):
        self._log_token(trainer, is_primary=is_primary, epoch=0)

    def on_epoch(
        self, trainer: "GradientDescentTrainer", epoch: int, is_primary: bool = True, **kwargs,
    ):
        self._log_token(trainer, epoch=epoch, is_primary=is_primary)

    def _make_it_human_readable(self, data, vocab: Vocabulary, vocab_namespace: str):
        if vocab_namespace is not None:
            data = tensor2tokens(data, vocab, vocab_namespace)
        return data

    def _log_token(self, trainer: GradientDescentTrainer, epoch: int, is_primary: bool) -> None:
        if not is_primary:
            pass
        logger.info(f"===== Sample at Epoch {epoch} =====")
        vocab = trainer.model.vocab

        # sample a instance
        # prevent data_loader to read all instances
        max_instances_in_memory = trainer.data_loader.max_instances_in_memory
        trainer.data_loader.max_instances_in_memory = 8
        batch = next(iter(trainer.data_loader))

        # log input tokens
        index = 0
        for signature, vocab_namespace in self.input_name_spaces.items():
            input_ = batch[signature]
            while isinstance(input_, dict):
                input_ = input_["tokens"]
            input_ = input_[index]
            human_redable_tokens = self._make_it_human_readable(input_, vocab, vocab_namespace)

            logger.info(f"{signature}({vocab_namespace}): {human_redable_tokens}")

        # log output tokens
        if self.output_name_spaces:
            model = trainer.model
            batch = move_to_device(batch, model.model_weight.device)
            output_dict = model(**batch)
            for signature, vocab_namespace in self.output_name_spaces.items():
                output = output_dict[signature][index]
                human_redable_tokens = self._make_it_human_readable(output, vocab, vocab_namespace)
                logger.info(f"{signature}({vocab_namespace}): {human_redable_tokens}")

            model.get_metrics(reset=True)

        trainer.data_loader.max_instances_in_memory = max_instances_in_memory
