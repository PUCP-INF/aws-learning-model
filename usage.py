from transformers import T5ForConditionalGeneration, T5Tokenizer, GenerationConfig

model_path = "./fine_tuned_t5_model"

tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path, device_map="auto")

model.eval()

prompt = f"Please answer this question: What is a bucket policy?"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

generated_ids = model.generate(
    input_ids=input_ids,
    generation_config=GenerationConfig(
        max_new_tokens=512,
        do_sample=True,
        top_p=0.9,
        top_k=0,
        temperature=0.1
    )
)
outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
for out in outputs:
    print(out)