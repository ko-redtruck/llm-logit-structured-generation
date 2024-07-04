from transformers import AutoTokenizer, T5ForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

# training
input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
outputs = model(input_ids=input_ids, labels=labels)
loss = outputs.loss
logits = outputs.logits

# inference
input_ids = tokenizer(
    ["summarize: studies have shown that owning a dog is good for you", "hello world"], return_tensors="pt", padding=True
)  # Batch size 1
outputs = model.generate(**input_ids, output_scores=True, return_dict_in_generate=True)
print(dir(outputs))
# studies have shown that owning a dog is good for you.