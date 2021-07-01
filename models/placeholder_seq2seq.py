from typing import Dict, Union, List
from overrides import overrides

import torch

from allennlp.models import Model
from allennlp.data import TextFieldTensors
from allennlp.nn import util
from models.utils import tensor_to_string_tokens
from .seq2seq import Seq2Seq

from models.metrics.target_translation import TargetTranslation
from models.dataset_reader.placeholder_seq2seq_reader import Placeholder

from lemminflect import getInflection

INFLECTION_TAGS = {"<NNS>", "<NN>", "<VB>", "<VBD>", "<VBG>", "<VBN>", "<VBP>", "<VBZ>", "LEMMA"}


def inflect_lemma(lemma: str, tag: str) -> str:
    if tag not in INFLECTION_TAGS:
        raise ValueError(f"You have provided an invalid inflection tag {tag}." f"Please choose from :{INFLECTION_TAGS}")
    prefix = lemma.split()[:-1]
    head = lemma.split()[-1]
    # always return the first one when it has two candidates (e.g., ["dogs", "dog"])
    inflected_head = getInflection(head, tag=tag[1:-1])[0]
    word = " ".join(prefix + [inflected_head])
    return word


@Model.register("placeholder_seq2seq")
class PlaceholderSeq2Seq(Seq2Seq):
    def __init__(
        self,
        specified_inflection: str = None,
        keep_placeholder: bool = False,
        use_source_factor: bool = False,
        evaluate_lemma: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.metrics["target_translation"] = TargetTranslation(evaluate_lemma=evaluate_lemma)
        self.specified_inflection = specified_inflection
        self.keep_placeholder = keep_placeholder
        self.use_source_factor = use_source_factor
        if self.use_source_factor:
            self.source_factor_embeddings = torch.nn.Embedding(
                num_embeddings=2, embedding_dim=self.source_embedder.get_output_dim()
            )

    def encode(
        self, source_tokens: TextFieldTensors, source_factor: torch.LongTensor = None
    ) -> Dict[str, torch.Tensor]:
        embedded_input = self.source_embedder(source_tokens)

        if self.use_source_factor:
            source_factor_embeddings = self.source_factor_embeddings(source_factor)
            embedded_input = embedded_input + source_factor_embeddings

        source_mask = util.get_text_field_mask(source_tokens)

        encoder_output = self.encoder(embedded_input, source_mask)
        return encoder_output

    @overrides
    def forward(
        self,
        source_tokens: TextFieldTensors,
        target_tokens: TextFieldTensors = None,
        start_tokens: Union[int, torch.Tensor] = None,
        placeholder: List[Placeholder] = None,
        source_factor: torch.LongTensor = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        encoder_output = self.encode(source_tokens, source_factor)
        batch_size = encoder_output["encoder_output"].size(0)

        if self.training:
            return self.fit_target_tokens(target_tokens, context=encoder_output)

        if target_tokens is not None:
            output_dict = self.evaluate_loss(target_tokens, context=encoder_output)
            start_tokens = target_tokens[self.target_token_namespace][self.target_token_namespace][:, 0]
        else:
            output_dict = {}

        prediction_dict = self.decode(start_tokens=start_tokens, context=encoder_output, batch_size=batch_size)
        prediction_tensor = prediction_dict["prediction_tensor"]
        prediction = self.postprocess(
            prediction_tensor, placeholder=placeholder, specified_inflection=self.specified_inflection,
        )
        output_dict.update({"prediction": prediction})

        if placeholder is not None:
            self.metrics["target_translation"](prediction, placeholder)

        if target_tokens is not None:
            target = self.postprocess(
                target_tokens[self.target_token_namespace][self.target_token_namespace][:, 1:], placeholder=placeholder,
            )
            output_dict.update({"target": target})

            if self.token_metrics is not None:
                for metric in self.token_metrics.values():
                    metric(prediction, target)

        output_dict.update(prediction_dict)

        return output_dict

    def postprocess(
        self, target_tensor: torch.Tensor, placeholder: List[Placeholder], specified_inflection: str = None,
    ) -> List[List[str]]:
        tokens = tensor_to_string_tokens(target_tensor, self.vocab, self.target_vocab_namespace, self.target_end_index,)

        if not self.keep_placeholder:
            tokens = self.replace_placeholders(tokens, placeholder, specified_inflection=specified_inflection)
        return tokens

    @staticmethod
    def replace_placeholders(
        tokens_list: List[List[str]], placeholder: List[Placeholder], specified_inflection: str = None,
    ) -> List[List[str]]:

        for i, (tokens, p) in enumerate(zip(tokens_list, placeholder)):
            if p.placeholder_token is None:
                continue
            for j, t in enumerate(tokens):
                if specified_inflection is None:
                    word = p.target_word
                elif specified_inflection == "LEMMA":
                    word = p.target_lemma
                else:
                    word = inflect_lemma(p.target_lemma, f"<{specified_inflection}>")
                tokens_list[i][j] = t.replace(p.placeholder_token, word)
        return tokens_list
