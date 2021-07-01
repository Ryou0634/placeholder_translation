from typing import Dict, Union, List, NamedTuple, Optional, Tuple
from overrides import overrides

import torch

from allennlp.models import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.data import TextFieldTensors
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN
from allennlp.training.metrics import Average
from allennlp.nn.util import get_text_field_mask
from allennlp.training.util import get_batch_size

from models.dataset_reader.placeholder_seq2seq_reader import Placeholder

from .utils import tensor_to_string_tokens, merge_embedding_sequences

from models.modules.sequence_encoders import SequenceEncoder
from models.modules.sequence_decoders import SequenceDecoder
from models.modules.submodules.transformer_submodules import PositionEmbedding
from .sequence_generator import SequenceGenerator

from .placeholder_seq2seq import PlaceholderSeq2Seq


class InflectionInstance(NamedTuple):
    batch_idx: int
    placeholder_idx: int
    lemma_tokens: List[str]
    target_tokens: Optional[List[str]]


def list_to_tensor(token_ids: List[List[int]], padding_value: int = 0) -> torch.Tensor:
    max_length = max([len(tokens) for tokens in token_ids])
    for tokens in token_ids:
        tokens += [padding_value] * (max_length - len(tokens))
    return torch.LongTensor(token_ids)


@Model.register("inflectional_placeholder_seq2seq")
class InflectionalPlaceholderSeq2Seq(PlaceholderSeq2Seq):
    def __init__(
        self,
        max_inflection_length: int,
        inflection_character_decoder: SequenceDecoder,
        inflection_context_encoder: SequenceEncoder = None,
        source_token_namespace: str = "tokens",
        character_namespace: str = "characters",
        only_train_character_decoder: bool = False,
        feed_all_context_with_enc: bool = False,
        use_double_attention: bool = False,
        no_type_embeddings: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.inflection_generator = SequenceGenerator(
            vocab=self.vocab,
            target_embedder=BasicTextFieldEmbedder(
                {
                    character_namespace: Embedding(
                        embedding_dim=inflection_character_decoder.get_output_dim(),
                        vocab_namespace=character_namespace,
                        vocab=self.vocab,
                    )
                }
            ),
            decoder=inflection_character_decoder,
            share_target_weights=True,
            max_decoding_length=max_inflection_length,
            target_vocab_namespace=character_namespace,
            target_token_namespace=character_namespace,
            beam_size=None,
        )

        if inflection_character_decoder.get_output_dim() != self.source_embedder.get_output_dim():
            self.tgt_to_char_dec_linear = torch.nn.Linear(
                self.target_embedder.get_output_dim(), inflection_character_decoder.get_output_dim()
            )
        else:
            self.tgt_to_char_dec_linear = None

        self.inflection_context_encoder = inflection_context_encoder

        self.lemma_position_embedding = PositionEmbedding(max_sequence_length=max_inflection_length)
        self.context_position_embedding = PositionEmbedding(max_sequence_length=self.max_decoding_length)

        self.character_namespace = character_namespace
        self.source_token_namespace = source_token_namespace
        self.feed_all_context_with_enc = feed_all_context_with_enc

        self.metrics["seq2seq_loss"] = Average()
        self.metrics["inflection_loss"] = Average()

        self.no_type_embeddings = no_type_embeddings
        self.only_train_character_decoder = only_train_character_decoder
        self.use_double_attention = use_double_attention
        if not self.use_double_attention:
            if self.no_type_embeddings:
                self.lemma_type_embedding = None
                self.hidden_type_embedding = None
            else:
                self.lemma_type_embedding = torch.nn.Parameter(
                    torch.randn(inflection_character_decoder.get_output_dim())
                )
                self.hidden_type_embedding = torch.nn.Parameter(
                    torch.randn(inflection_character_decoder.get_output_dim())
                )

    @overrides
    def forward(
        self,
        source_tokens: TextFieldTensors,
        target_tokens: TextFieldTensors = None,
        start_tokens: Union[int, torch.Tensor] = None,
        placeholder: List[Placeholder] = None,
        context: Dict[str, torch.Tensor] = None,
        lemma_characters: TextFieldTensors = None,
        inflection_characters: TextFieldTensors = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        batch_size = get_batch_size(source_tokens)
        if self.training:

            if self.only_train_character_decoder:
                target_mask = get_text_field_mask(target_tokens)[:, 1:].contiguous()
                output_dict = {
                    "loss": 0,
                    "prediction_tensor": target_tokens[self.target_token_namespace][self.target_token_namespace][:, 1:],
                    "target_mask": target_mask,
                }
            else:
                encoder_output = self.encode(source_tokens)
                output_dict = self.fit_target_tokens(target_tokens, context=encoder_output)
                self.metrics["seq2seq_loss"](output_dict["loss"].item())

            fit_inflection_output = self.fit_inflection(
                target_tensor=target_tokens[self.target_token_namespace][self.target_token_namespace][:, 1:],
                placeholder=placeholder,
                target_mask=output_dict["target_mask"],
                lemma_characters=lemma_characters,
                inflection_characters=inflection_characters,
            )

            if fit_inflection_output:
                self.metrics["inflection_loss"](fit_inflection_output["loss"].item())
                output_dict["loss"] += fit_inflection_output["loss"]
            return output_dict

        encoder_output = self.encode(source_tokens)
        if target_tokens is not None:
            output_dict = self.evaluate_loss(target_tokens, context=encoder_output)
            fit_inflection_output = self.fit_inflection(
                target_tensor=target_tokens[self.target_token_namespace][self.target_token_namespace][:, 1:],
                placeholder=placeholder,
                target_mask=output_dict["target_mask"],
                lemma_characters=lemma_characters,
                inflection_characters=inflection_characters,
            )
            self.metrics["seq2seq_loss"](output_dict["loss"].item())
            if fit_inflection_output is not None:
                self.metrics["inflection_loss"](fit_inflection_output["loss"].item())
                output_dict["loss"] += fit_inflection_output["loss"]
                for k, v in fit_inflection_output.items():
                    output_dict["fit_inflector_" + k] = v
            start_tokens = target_tokens[self.target_token_namespace][self.target_token_namespace][:, 0]
        else:
            output_dict = {}

        decoder_output_dict = self.decode(start_tokens=start_tokens, context=encoder_output, batch_size=batch_size)
        # perform inflection prediction
        predicted_inflection_output = self.predict_inflection(
            target_tensor=decoder_output_dict["prediction_tensor"],
            placeholder=placeholder,
            target_mask=decoder_output_dict["target_mask"],
            lemma_characters=lemma_characters,
        )

        prediction = self.postprocess(
            decoder_output_dict["prediction_tensor"], placeholder, predicted_inflection_output
        )

        if predicted_inflection_output is not None:
            for k, v in predicted_inflection_output.items():
                output_dict["inflector_" + k] = v

        output_dict.update({"prediction": prediction})

        if placeholder is not None:
            self.metrics["target_translation"](prediction, placeholder)

        if target_tokens is not None:
            target = self.postprocess(
                target_tokens[self.target_token_namespace][self.target_token_namespace][:, 1:], placeholder
            )
            output_dict.update({"target": target})
            if self.token_metrics is not None:
                for metric in self.token_metrics.values():
                    metric(prediction, target)

        output_dict.update(decoder_output_dict)

        return output_dict

    def postprocess(
        self, target_tensor: torch.Tensor, placeholder: List[Placeholder], prediction_output=None,
    ) -> List[List[str]]:
        tokens = tensor_to_string_tokens(target_tensor, self.vocab, self.target_vocab_namespace, self.target_end_index)

        if self.keep_placeholder:
            return tokens

        if prediction_output is not None:
            predicted_placeholder = []
            for idx, p in enumerate(placeholder):
                if idx in prediction_output["predicted_index"]:
                    predicted_word = "".join(
                        prediction_output["prediction"][prediction_output["predicted_index"].index(idx)]
                    )
                    if len(predicted_word) > 0 and predicted_word[0] == "▁":
                        predicted_word = predicted_word[1:]
                    predicted_placeholder.append(
                        Placeholder(target_word=predicted_word, placeholder_token=p.placeholder_token)
                    )
                else:
                    predicted_placeholder.append(p)

            tokens = self.replace_placeholders(tokens, predicted_placeholder)
        else:
            tokens = self.replace_placeholders(tokens, placeholder)
        return tokens

    def _prepare_inflection_instances(
        self, target_tensor: torch.Tensor, placeholder: List[Placeholder]
    ) -> List[InflectionInstance]:
        tokens = tensor_to_string_tokens(target_tensor, self.vocab, self.target_vocab_namespace, self.target_end_index)

        # get PLACEHOLDER index
        inflection_instances = []
        for batch_idx, (ts, p) in enumerate(zip(tokens, placeholder)):
            for i, t in enumerate(ts):
                if t == p.placeholder_token:
                    inflection_instances.append(
                        InflectionInstance(batch_idx, i, p.target_lemma_tokens, p.target_word_tokens)
                    )
        if len(inflection_instances) == 0:
            return None

        return inflection_instances

    def _prepare_attention_state(
        self,
        inflection_instances: List[InflectionInstance],
        context_tensors: torch.Tensor,
        target_mask: torch.BoolTensor,
        lemma_characters: TextFieldTensors = None,
    ):
        batch_indices = [i.batch_idx for i in inflection_instances]

        if lemma_characters is not None:
            lemma_embeddings = self.inflection_generator.target_embedder(lemma_characters)[batch_indices]
            lemma_mask = get_text_field_mask(lemma_characters)[batch_indices]
        else:
            lemma_embeddings, lemma_mask = self.make_lemma_subword_embeddings(inflection_instances)

        lemma_embeddings = self.lemma_position_embedding(lemma_embeddings)

        context_tensors = context_tensors[batch_indices]
        target_mask = target_mask[batch_indices]

        if self.use_double_attention:
            state = {
                "encoder_output": lemma_embeddings,
                "source_mask": lemma_mask,
                "second_encoder_output": context_tensors,
                "second_source_mask": target_mask,
            }
        else:
            if not self.no_type_embeddings:
                lemma_embeddings = lemma_embeddings + self.lemma_type_embedding
                context_tensors = context_tensors + self.hidden_type_embedding
            merged_embeddings, merged_mask = merge_embedding_sequences(
                lemma_embeddings, lemma_mask, context_tensors, target_mask
            )
            state = {"encoder_output": merged_embeddings, "source_mask": merged_mask}
        return state

    def _prepare_context_tensors(
        self,
        target_tensor: torch.LongTensor,
        target_mask: torch.BoolTensor,
        inflection_instances: List[InflectionInstance],
    ):
        context_tensors = self.target_embedder({"tokens": {"tokens": target_tensor}})
        if self.tgt_to_char_dec_linear:
            context_tensors = self.tgt_to_char_dec_linear(context_tensors)

        if self.inflection_context_encoder:
            encoder_output = self.inflection_context_encoder.forward(context_tensors, mask=target_mask)
            context_tensors = encoder_output["encoder_output"]
            if not self.feed_all_context_with_enc:
                placeholder_position_indices = [0] * len(context_tensors)
                for i in inflection_instances:
                    placeholder_position_indices[i.batch_idx] = i.placeholder_idx
                batch_range = [i for i in range(len(context_tensors))]
                context_tensors = context_tensors[batch_range, placeholder_position_indices].unsqueeze(dim=1)
                target_mask = target_mask[batch_range, placeholder_position_indices].unsqueeze(dim=1)
        else:
            context_tensors = self.context_position_embedding(context_tensors)
        return context_tensors, target_mask

    def fit_inflection(
        self,
        target_tensor: torch.Tensor,
        placeholder: List[Placeholder],
        target_mask: torch.BoolTensor,
        lemma_characters: TextFieldTensors = None,
        inflection_characters: TextFieldTensors = None,
    ):

        inflection_instances = self._prepare_inflection_instances(target_tensor, placeholder)
        if inflection_instances is None:
            return None

        context_tensors, target_mask = self._prepare_context_tensors(target_tensor, target_mask, inflection_instances)

        state = self._prepare_attention_state(inflection_instances, context_tensors, target_mask, lemma_characters)

        if inflection_characters is not None:
            batch_indices = [i.batch_idx for i in inflection_instances]
            character_tensor = inflection_characters[self.character_namespace]["tokens"][batch_indices]
            target_text_field_tensor = {self.character_namespace: {"tokens": character_tensor}}
        else:
            target_text_field_tensor = self.make_output_batch(inflection_instances)
        output_dict = self.inflection_generator.fit_target_tokens(target_text_field_tensor, state)
        output_dict["predicted_index"] = [i.batch_idx for i in inflection_instances]

        return output_dict

    def _prepare_encoder_context(
        self, context_tensors: torch.Tensor, target_mask: torch.Tensor, inflection_instances: List[InflectionInstance]
    ):
        encoder_output = self.inflection_context_encoder.forward(context_tensors, mask=target_mask)
        context_tensors = encoder_output["encoder_output"]
        if not self.feed_all_context_with_enc:
            placeholder_position_indices = [0] * len(context_tensors)
            for i in inflection_instances:
                placeholder_position_indices[i.batch_idx] = i.placeholder_idx
            batch_range = [i for i in range(len(context_tensors))]
            context_tensors = context_tensors[batch_range, placeholder_position_indices].unsqueeze(dim=1)
            target_mask = target_mask[batch_range, placeholder_position_indices].unsqueeze(dim=1)
        return context_tensors, target_mask

    def predict_inflection(
        self,
        target_tensor: torch.LongTensor,
        placeholder: List[Placeholder],
        target_mask: torch.BoolTensor,
        lemma_characters: TextFieldTensors = None,
    ):
        inflection_instances = self._prepare_inflection_instances(target_tensor, placeholder)
        if inflection_instances is None:
            return None

        context_tensors, target_mask = self._prepare_context_tensors(target_tensor, target_mask, inflection_instances)

        state = self._prepare_attention_state(inflection_instances, context_tensors, target_mask, lemma_characters)

        start_tokens = [
            self.vocab.get_token_index(placeholder[i.batch_idx].placeholder_token, namespace=self.character_namespace)
            for i in inflection_instances
        ]
        start_tokens = torch.LongTensor(start_tokens).to(target_mask.device)
        output_dict = self.inflection_generator.decode(
            context=state, batch_size=len(inflection_instances), start_tokens=start_tokens
        )
        output_dict.update({"prediction": self.inflection_generator.postprocess(output_dict["prediction_tensor"])})

        output_dict["predicted_index"] = [i.batch_idx for i in inflection_instances]

        return output_dict

    def make_lemma_subword_embeddings(
        self, inflection_instances: List[InflectionInstance]
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
        lemma_token_ids = []
        for instance in inflection_instances:
            lemma_ids = [
                self.vocab.get_token_index(t, namespace=self.target_vocab_namespace) for t in instance.lemma_tokens
            ]
            lemma_token_ids.append(lemma_ids)

        padding_idx = self.vocab.get_token_index(DEFAULT_PADDING_TOKEN, namespace=self.target_vocab_namespace)
        input_token_tensor = list_to_tensor(lemma_token_ids, padding_idx).to(self.device)
        source_text_field_tensor = {self.source_token_namespace: {self.source_token_namespace: input_token_tensor}}
        lemma_embeddings = self.source_embedder(source_text_field_tensor)

        # get mask
        source_mask = input_token_tensor != padding_idx
        return lemma_embeddings, source_mask

    def make_output_batch(self, inflection_instances: List[InflectionInstance]) -> TextFieldTensors:
        output_token_ids = []
        for instance in inflection_instances:

            out_ids = [
                self.vocab.get_token_index(t, namespace=self.target_vocab_namespace) for t in instance.target_tokens
            ]
            out_ids = [self.target_start_index] + out_ids + [self.target_end_index]
            output_token_ids.append(out_ids)

        padding_idx = self.vocab.get_token_index(DEFAULT_PADDING_TOKEN, namespace=self.target_vocab_namespace)
        output_token_tensor = list_to_tensor(output_token_ids, padding_idx).to(self.device)

        target_text_field_tensor = {self.target_token_namespace: {self.target_token_namespace: output_token_tensor}}

        return target_text_field_tensor
