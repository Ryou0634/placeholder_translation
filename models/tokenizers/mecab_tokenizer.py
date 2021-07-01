from typing import List, Optional

from allennlp.data import Token
from allennlp.data.tokenizers import Tokenizer
import unicodedata


def parse_feature_for_ipadic(elem) -> Token:
    surface, feature = elem.split("\t")
    (postag, postag2, postag3, postag4, inflection, conjugation, base_form, *other,) = feature.split(",")

    # For words not in a dictionary
    if len(other) == 2:
        yomi, pron = other
    else:
        yomi, pron = None, None

    return Token(text=surface, lemma_=base_form, pos_=postag, tag_=postag2)


@Tokenizer.register("mecab")
class MeCabTokenizer(Tokenizer):
    """Wrapper class forexternal text analyzers"""

    def __init__(
        self,
        user_dictionary_path: Optional[str] = None,
        system_dictionary_path: Optional[str] = None,
        dictionary_format: Optional[str] = None,
        with_pos_tag: bool = False,
        nfkc_normalize: bool = True,
        unnormalize_special_characters: bool = False
    ) -> None:
        """
        Initializer for MeCabTokenizer.
        Parameters
        ---
        dictionary_path (Optional[str]=None)
            path to a custom dictionary (option)
            it is used by `mecab -u [dictionary_path]`
        with_pos_tag (bool=False)
            flag determines ifkonoha.tokenizer include pos tags.
        """
        import natto

        self.with_pos_tag = with_pos_tag

        flag = ""

        if not self.with_pos_tag:
            flag += " -Owakati"

        if isinstance(user_dictionary_path, str):
            flag += " -u {}".format(user_dictionary_path)

        if isinstance(system_dictionary_path, str):
            flag += " -d {}".format(system_dictionary_path)

        self._tokenizer = natto.MeCab(flag)

        # If dictionary format is not specified,
        # konoha detects it by checking a name of system dictionary.
        # For instance, system_dictionary_path=mecab-ipadic-xxxx -> ipadic and
        #               system_dictionary_path=mecab-unidic-xxxx -> unidic.
        # If system_dictionary_path and dictionary_format are not given,
        # konoha assumes it uses mecab-ipadic (de facto standard).
        # Currently, konoha only supports ipadic. (TODO: unidic)
        if dictionary_format is None:
            if system_dictionary_path is None or "ipadic" in system_dictionary_path.lower():
                self._parse_feature = parse_feature_for_ipadic
            else:
                raise ValueError(f"Unsupported system dictionary: {system_dictionary_path}")

        else:
            if "ipadic" == dictionary_format.lower():
                self._parse_feature = parse_feature_for_ipadic
            else:
                raise ValueError(f"Unsupported dictionary format: {dictionary_format}")
        self.nfkc_normalize = nfkc_normalize

        self.unnormalize_special_characters = unnormalize_special_characters
        # Mecab produces incorrect parse when parsing Han-kaku special characters.
        # So we make it back to Zenkaku after NFKC normalization.
        self.trans_table = str.maketrans({"(": "（", ")": "）", "!": "！", "?": "？"})

    def tokenize(self, text: str) -> List[Token]:
        """Tokenize"""

        if self.nfkc_normalize:
            text = unicodedata.normalize("NFKC", text)
            if self.unnormalize_special_characters:
                text = text.translate(self.trans_table)

        return_result = []
        parse_result = self._tokenizer.parse(text).rstrip(" ")
        if self.with_pos_tag:
            for elem in parse_result.split("\n")[:-1]:
                return_result.append(self._parse_feature(elem))
        else:
            for surface in parse_result.split(" "):
                return_result.append(Token(surface))
        return return_result
