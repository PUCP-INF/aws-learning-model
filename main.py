import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# T5ForQuestionAnswering
# T5ForTokenClassification
# T5ForSequenceClassification
# T5ForConditionalGeneration

has_cuda = torch.cuda.is_available()

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map="auto")

input_text = "translate English to German: How old are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

if has_cuda:
    input_ids = input_ids.to("cuda")

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))