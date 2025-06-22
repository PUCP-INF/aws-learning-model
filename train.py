import json
import sys

from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq

from datasets import Dataset

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

with open("./corpus/corpus2.json", "r") as f:
    raw_data = json.load(f)

dataset = Dataset.from_list(raw_data)

def preprocess_function(data):
    prefix = "Please answer this question: "
    inputs = [prefix + q for q in data["question"]]
    model_inputs = tokenizer(inputs)

    labels = tokenizer(text_target=data["answer"])

    model_inputs["labels"] = labels.input_ids
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

#split_dataset = tokenized_dataset.train_test_split(test_size=0.2, shuffle=True)

training_args = TrainingArguments(
    output_dir="./results",
    # evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    # per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    # bf16=True,
    torch_compile=True
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    # eval_dataset=split_dataset["test"],
    data_collator=data_collator
)

# Fine-tune the model
trainer.train()

# Evaluate the model on the validation dataset
# eval_results = trainer.evaluate()

# Print the evaluation results
# print(f"Evaluation results: {eval_results}")

output_model_path = "./fine_tuned_t5_model"
trainer.save_model(output_model_path)

print(f"Model saved to {output_model_path}")