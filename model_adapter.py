import torch
from transformers import AutoModel, PreTrainedModel

class ModelAdapter:
    def generate(self, token_ids: list[list[int]]) -> list[list[torch.tensor]]:
        raise NotImplementedError()
    
class HuggingFaceModelAdapter(ModelAdapter):
    def __init__(self, model: PreTrainedModel) -> None:
        super().__init__()
        
        self.model = model

    @classmethod
    def from_pretrained(cls, model_id: str):
        model = AutoModel.from_pretrained(model_id)
        return cls(model)
    
    def generate(self, token_ids: list[list[int]]) -> list[list[torch.tensor]]: #(Batch, sequence_length, vocab_size)
        outputs = self.model.generate(token_ids, output_scores=True, return_dict_in_generate=True)
        # Shape: (SEQUENCE_LENGTH, Batch_size, vocab_size). Over the sequence length it is a dict
        raw_logits = outputs.scores

        batch_size = len(raw_logits[0])
        seq_length = len(raw_logits)

        logits = [[raw_logits[j][i] for j in range(seq_length)] for i in range(batch_size)]
        return logits
    
        

    