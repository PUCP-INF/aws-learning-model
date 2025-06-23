from transformers import T5ForConditionalGeneration, T5Tokenizer, GenerationConfig

model_path = "./fine_tuned_t5_model"

tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path, device_map="auto")

model.eval()

prompt = f"answer this question: What do objects in Amazon S3 consist of?"
input_ids = tokenizer(
    prompt,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512
).input_ids.to("cuda")

generated_ids = model.generate(
    input_ids=input_ids,
    generation_config=GenerationConfig(
        max_new_tokens=512,
        num_beams=1,
        do_sample=True,
        top_p=0.9,
        top_k=35,
        temperature=0.2
    )
)
outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(outputs[0])