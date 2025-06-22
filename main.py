import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# T5ForQuestionAnswering
# T5ForTokenClassification
# T5ForSequenceClassification
# T5ForConditionalGeneration

has_cuda = torch.cuda.is_available()

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map="auto")

while True:
    user_question = input("Enter your question: ")
    prompt = f"Answer this question: {user_question}"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    if has_cuda:
        input_ids = input_ids.to("cuda")

    generated_ids = model.generate(input_ids)
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    for out in outputs:
        print(out)
