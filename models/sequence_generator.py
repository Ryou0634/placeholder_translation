from typing import Dict, List, Tuple, Union
from overrides import overrides
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.modules import TextFieldEmbedder, Embedding
from allennlp.models.model import Model
from allennlp.nn.util import get_text_field_mask
from allennlp.nn.beam_search import BeamSearch
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics import Metric

from models.modules.sequence_decoders import SequenceDecoder
from models.metrics.loss_to_perplexity import LossToPerplexity

from models.utils import copy_tensor_dict, tensor_to_string_tokens
from models.modules.sampling import sampling_from_logits
from models.modules.sequence_cross_entropy import SequenceCrossEntropy


logger = logging.getLogger(__name__)


@Model.register("sequence_generator")
class SequenceGenerator(Model):
    """
    Sequence generator or language model.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        target_embedder: TextFieldEmbedder,
        decoder: SequenceDecoder,
        sequence_loss: SequenceCrossEntropy = SequenceCrossEntropy(),
        share_target_weights: bool = False,
        max_decoding_length: int = 100,
        beam_size: int = None,
        target_vocab_namespace: str = "tokens",
        target_token_namespace: str = "tokens",
        token_metrics: Dict[str, Metric] = None,
        initializer: InitializerApplicator = None,
    ):

        super().__init__(vocab)

        self.target_embedder = target_embedder

        self.decoder = decoder
        self.target_vocab_namespace = target_vocab_namespace

        self.target_token_namespace = target_token_namespace

        self.output_projection_layer = nn.Linear(
            decoder.get_output_dim(), self.vocab.get_vocab_size(self.target_vocab_namespace)
        )

        if share_target_weights:
            token_embedding = self.target_embedder._token_embedders[self.target_token_namespace]
            if not isinstance(token_embedding, Embedding):
                token_embedding = token_embedding._embedding._module

            if self.output_projection_layer.weight.shape != token_embedding.weight.shape:
                raise Exception(
                    "You cannot enable share_target_weights.\n"
                    f"{self.output_projection_layer.weight.shape} != {token_embedding.weight.shape}"
                )
            self.output_projection_layer.weight = token_embedding.weight

        self.target_start_index = self.vocab.get_token_index(START_SYMBOL, self.target_vocab_namespace)
        self.target_end_index = self.vocab.get_token_index(END_SYMBOL, self.target_vocab_namespace)

        self.max_decoding_length = max_decoding_length
        self.beam_size = beam_size
        if beam_size is not None:
            self.beam_search = BeamSearch(self.target_end_index, max_steps=max_decoding_length, beam_size=beam_size)
        else:
            self.beam_search = None

        self.sequence_loss = sequence_loss

        self.metrics = {
            "perplexity": LossToPerplexity(),
        }

        self.token_metrics = token_metrics

        if initializer:
            initializer(self)

    @property
    def model_weight(self) -> torch.Tensor:
        "Used to create new tensor"
        return self.output_projection_layer.weight

    @property
    def device(self) -> torch.device:
        return self.model_weight.device

    @overrides
    def forward(
        self,
        target_tokens: TextFieldTensors = None,
        start_tokens: Union[int, torch.Tensor] = None,
        context: Dict[str, torch.Tensor] = None,
        batch_size: int = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        if self.training:
            return self.fit_target_tokens(target_tokens, context=context)

        if target_tokens is not None:
            output_dict = self.evaluate_loss(target_tokens, context=context)
            start_tokens = target_tokens[self.target_token_namespace][self.target_token_namespace][:, 0]
        else:
            output_dict = {}

        prediction_dict = self.decode(start_tokens=start_tokens, context=context, batch_size=batch_size)
        prediction_tensor = prediction_dict["prediction_tensor"]
        prediction = self.postprocess(prediction_tensor, **kwargs)
        output_dict.update({"prediction": prediction})

        if target_tokens is not None:
            target = self.postprocess(
                target_tokens[self.target_token_namespace][self.target_token_namespace][:, 1:], **kwargs
            )
            output_dict.update({"target": target})
            if self.token_metrics is not None:
                for metric in self.token_metrics.values():
                    metric(prediction, target)

        output_dict.update(prediction_dict)

        return output_dict

    def evaluate_loss(self, target_tokens: TextFieldTensors, context: Dict[str, torch.Tensor] = None):
        if context is not None:
            context = copy_tensor_dict(context)
        return self.fit_target_tokens(target_tokens, context=context)

    def fit_target_tokens(self, target_tokens: TextFieldTensors, context: Dict[str, torch.Tensor] = None):
        """
        Process the tokens across the time steps at once.
        """
        # shape: (batch_size, seq_len, embed_size)
        target_embeddings = self.target_embedder(target_tokens)
        target_embeddings = target_embeddings[:, :-1]  # the last is either @end@ or @padding@

        target_mask = get_text_field_mask(target_tokens)

        # shape: (batch_size, seq_len, hidden_size)
        decoder_outputs, state = self.decoder.forward_across_time_steps(target_embeddings, context)

        logits = self.output_projection_layer(decoder_outputs)  # shape: (batch_size, seq_len, n_class)

        output_dict = self.sequence_loss(
            logits,
            target_tokens[self.target_token_namespace]["tokens"][:, 1:].contiguous(),
            target_mask[:, 1:].contiguous(),
        )

        _, prediction = logits.max(dim=2)

        self.metrics["perplexity"](output_dict["loss"].item())
        output_dict["decoder_outputs"] = decoder_outputs
        output_dict["prediction_tensor"] = prediction
        output_dict["target_mask"] = target_mask[:, 1:].contiguous()

        if state:
            output_dict.update(state)
        return output_dict

    @torch.no_grad()
    def decode_loop(
        self,
        start_tokens: Union[int, torch.Tensor] = None,
        context: Dict[str, torch.tensor] = None,
        batch_size: int = 1,
        sampling_temperature: float = 0.0,
        sampling_top_k: int = None,
        sampling_top_p: float = None,
    ):
        context = context or {}

        if start_tokens is None:
            start_tokens = self.target_start_index

        if isinstance(start_tokens, int):
            start_tokens = self.model_weight.new_full((batch_size,), fill_value=start_tokens, dtype=torch.long)

        last_predictions = start_tokens
        state = context

        step_logits: List[torch.Tensor] = []
        step_predictions: List[torch.Tensor] = []
        step_decoder_outputs: List[torch.Tensor] = []
        step_target_mask: List[torch.Tensor] = []
        has_produced_eos = last_predictions.new_zeros(batch_size).bool()
        for timestep in range(self.max_decoding_length):
            decoder_input = last_predictions

            embedded_input = self.target_embedder(
                {self.target_token_namespace: {self.target_token_namespace: decoder_input}}
            )

            decoder_output, state = self.decoder(embedded_input, state)
            # list of tensors, shape: (batch_size, decoder_output_size)
            step_decoder_outputs.append(decoder_output)

            logits = self.output_projection_layer(decoder_output)
            # list of tensors, shape: (batch_size, num_classes)
            step_logits.append(logits)

            # shape (predicted_classes): (batch_size,)
            if sampling_temperature > 0:
                predicted_classes = sampling_from_logits(
                    logits, sampling_temperature, top_k=sampling_top_k, top_p=sampling_top_p
                )
            else:
                _, predicted_classes = logits.max(dim=1)

            # shape (predicted_classes): (batch_size,)
            last_predictions = predicted_classes

            # list of tensors, shape: (batch_size,)
            step_predictions.append(last_predictions)

            # update end flags and if all instance has produced EOS, end the loop
            is_eos_token = last_predictions == self.target_end_index
            has_produced_eos = has_produced_eos | is_eos_token
            step_target_mask.append(has_produced_eos)
            if has_produced_eos.all():
                break

        return {
            "logits": torch.stack(step_logits, dim=1),
            "prediction_tensor": torch.stack(step_predictions, dim=1),
            "decoder_outputs": torch.stack(step_decoder_outputs, dim=1),
            "target_mask": ~torch.stack(step_target_mask, dim=1),
        }

    @torch.no_grad()
    def decode_beam_search(
        self,
        start_tokens: Union[int, torch.Tensor] = None,
        context: Dict[str, torch.Tensor] = None,
        batch_size: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """Make forward pass during prediction using a beam search."""
        context = context or {}
        assert "loss" not in context
        if start_tokens is None:
            start_tokens = self.target_start_index

        if isinstance(start_tokens, int):
            start_tokens = self.model_weight.new_full((batch_size,), fill_value=start_tokens, dtype=torch.long)
        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        all_top_k_predictions, log_probabilities = self.beam_search.search(start_tokens, context, self._take_step)

        # Beam search gives us the top-k results for each source sentence in the batch
        # but we just want the single best.
        output_dict = context
        output_dict["prediction_tensor"] = all_top_k_predictions[:, 0]

        return output_dict

    def decode(
        self,
        start_tokens: Union[int, torch.Tensor] = None,
        context: Dict[str, torch.Tensor] = None,
        batch_size: int = 1,
    ) -> Dict[str, torch.Tensor]:
        if self.beam_search:
            output_dict = self.decode_beam_search(start_tokens, context, batch_size)
        else:
            output_dict = self.decode_loop(start_tokens, context, batch_size)

        return output_dict

    def postprocess(self, target_tensor: torch.Tensor, **kwargs) -> List[List[str]]:
        """
        Convert tensor ids to string tokens.
        You can also replace tags when you overrides this.
        """
        tokens = tensor_to_string_tokens(target_tensor, self.vocab, self.target_vocab_namespace, self.target_end_index)
        return tokens

    def _take_step(
        self, last_predictions: torch.LongTensor, state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take a decoding step. This is called by the beam search class.

        Parameters
        ----------
        last_predictions : ``torch.Tensor``
            A tensor of shape ``(group_size,)``, which gives the indices of the prediction
            during the last time step.
        state : ``Dict[str, torch.Tensor]``
            A dictionary of tensors that contain the current state information
            needed to predict the next step, which includes the encoder outputs,
            the source mask, and the decoder hidden state and context. Each of these
            tensors has shape ``(group_size, *)``, where ``*`` can be any other number
            of dimensions.

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            A tuple of ``(log_probabilities, updated_state)``, where ``log_probabilities``
            is a tensor of shape ``(group_size, num_classes)`` containing the predicted
            log probability of each class for the next step, for each item in the group,
            while ``updated_state`` is a dictionary of tensors containing the encoder outputs,
            source mask, and updated decoder hidden state and context.

        Notes
        -----
            We treat the inputs as a batch, even though ``group_size`` is not necessarily
            equal to ``batch_size``, since the group may contain multiple states
            for each source sentence in the batch.
        """

        embedded_input = self.target_embedder(
            {self.target_token_namespace: {self.target_token_namespace: last_predictions}}
        )

        decoder_output, state = self.decoder(embedded_input, state)

        output_projection = self.output_projection_layer(decoder_output)

        # shape: (group_size, num_classes)
        class_log_probabilities = F.log_softmax(output_projection, dim=-1)

        return class_log_probabilities, state

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        output_dict = {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
        if self.token_metrics is not None:
            output_dict.update(
                {metric_name: metric.get_metric(reset) for metric_name, metric in self.token_metrics.items()}
            )

        return output_dict
