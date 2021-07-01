from allennlp.common import Registrable
from allennlp.data import Instance


class InstanceFilterer(Registrable):
    def to_filter(self, instance: Instance) -> bool:
        raise NotImplementedError


@InstanceFilterer.register("length")
class LengthFilterer(Instance):
    def __init__(self, field_name: str, max_length: None, min_length: None):
        if not max_length and not min_length:
            raise ValueError("Either max_length or min_length has to be not None.")

        self.field_name = field_name
        self.max_length = max_length
        self.min_length = min_length

    def to_filter(self, instance: Instance) -> bool:

        if self.max_length and len(instance[self.field_name]) > self.max_length:
            return True

        if self.min_length and len(instance[self.field_name]) < self.min_length:
            return True

        return False


@InstanceFilterer.register("ratio")
class LengthRatioFilterer(Instance):
    def __init__(self, source_field_name: str, target_field_name: str, max_ratio: None, min_ratio: None):
        if not max_ratio and not min_ratio:
            raise ValueError("Either max_ratio or min_ratio has to be not None.")

        self.source_field_name = source_field_name
        self.target_field_name = target_field_name
        self.max_ratio = max_ratio
        self.min_ratio = min_ratio

    def to_filter(self, instance: Instance) -> bool:

        ratio = len(instance[self.source_field_name]) / len(instance[self.target_field_name])

        if self.max_ratio and ratio > self.max_ratio:
            return True

        if self.min_ratio and ratio < self.min_ratio:
            return True

        return False
