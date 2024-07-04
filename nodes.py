from typing import Self
import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from model_adapter import ModelAdapter

class TokenNode:
    previous: Self = None
    nexts: dict[int, Self] = None
    logits: torch.Tensor = None
    token_id: int = None
    model: ModelAdapter = None

    is_encoder_end = False

    def __init__(self, previous: Self, token_id: int) -> None:
        self.nexts = {}
        self.previous = previous
        self.token_id = token_id

    @classmethod
    def from_text_without_logits(cls, text, tokenizer: PreTrainedTokenizerBase, is_encoder_sequence=False):
        token_ids = tokenizer.encode(text)
        assert len(token_ids) > 0, "Decoded an empty string"

        root = StartTokenNode()
        previous = root
        for token_id in token_ids:
            node = cls(previous, token_id)
            previous.nexts[token_id] = node
            previous = node
            
        if is_encoder_sequence:
            node.is_encoder_end = True

        return root

    def is_terminating_token(self) -> bool:
        return len(self.nexts) == 0
    
    def is_start_token_node(self) -> bool:
        return False
    
    def is_encoder_input_end(self) -> bool:
        return self.is_encoder_end
    
    def get_next_token(self) -> Self:
        assert len(self.nexts) == 1, "Calling get_next_token only is supported if there is only one possible path"
        return next(iter(self.nexts.values()))

    def add_text_without_logits(self, text: str, tokenizer: PreTrainedTokenizerBase) -> Self:
        new_path = TokenNode.from_text_without_logits(text, tokenizer)
        if new_path.is_start_token_node():
            new_path = new_path.get_next_token()
        
        node = self
        while new_path.token_id in node.nexts:
            if new_path.is_terminating_token():
                return self    
            node = node.nexts[new_path.token_id]
            new_path = new_path.get_next_token()
            


        node.nexts[new_path.token_id] = new_path
        new_path.previous = self
        return self

    def gather_all_sequences(self, include_self=True) -> list[list[int]]:
        sequences = []
        if len(self.nexts) == 0:
            if include_self:
                return [[self.token_id]]
            else:
                return [[]]

        for next_node in self.nexts.values():
            for sequence in next_node.gather_all_sequences():
                new_sequence = [self.token_id] + sequence if include_self else sequence
                sequences.append(new_sequence)

        return sequences
    
    def gather_encoder_sequence(self, include_self=True) -> list[int]:
        if self.is_encoder_input_end():
            return [self.token_id] if include_self else []
        else:
            return ([self.token_id] if include_self else []) + self.get_next_token().gather_encoder_sequence()
        
    def find_encoder_input_end_node(self) -> Self:
        if self.is_encoder_input_end():
            return self
        else:
            return self.get_next_token().find_encoder_input_end_node()

     
    def gather_all_encoder_decoder_sequence_pairs(self) -> tuple[list[list[int]], list[list[int]]]:
        encoder_sequence = self.gather_encoder_sequence()
        encoder_input_end_node = self.find_encoder_input_end_node()

        decoder_sequeunces = encoder_input_end_node.gather_all_sequences(include_self=False)
        encoder_sequeunces = [encoder_sequence for _ in range(len(decoder_sequeunces))]
        return encoder_sequeunces, decoder_sequeunces


class EmptyTokenNode(TokenNode):
    def __init__(self, previous: TokenNode, token_id: int) -> None:
        super().__init__()
        self.previous = previous
        self.token_id = token_id
        
class StartTokenNode(TokenNode):
    def __init__(self, ) -> None:
        super().__init__(None, None)

    def is_start_token_node(self) -> bool:
        return True
    
    def gather_encoder_sequence(self) -> list[int]:
        return super().gather_encoder_sequence(include_self=False)
    
    def gather_all_sequences(self) -> list[list[int]]:
        return super().gather_all_sequences(include_self=False)

class EncoderInputEndeTokenNode(TokenNode):
    def is_encoder_input_end(self) -> bool:
        return True
    
class DenseLogitsTokenNode(TokenNode):
    def __init__(self, previous: TokenNode, token_id: int, logits:  torch.Tensor = None) -> None:
        super().__init__(previous, token_id)
        self.logits = logits

    
    def compute(self, model: ModelAdapter=None):
        model = model if model is not None else self.model
        assert model is not None, "You need to either pass a model or set the model as an instance attribute"



