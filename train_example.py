import sys

from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

# Initialize the T5 tokenizer and model (T5-small in this case)
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

# Preprocessing the dataset: Prepare input-output pairs for T5
def preprocess_function(examples):
    inputs = [f"Question: {question}  Passage: {passage}" for question, passage in
              zip(examples['question'], examples['passage'])]
    targets = ['true' if answer else 'false' for answer in examples['answer']]

    # Tokenize inputs and outputs
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
    labels = tokenizer(targets, max_length=10, truncation=True, padding='max_length')
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

# Load the BoolQ dataset
dataset = load_dataset("yahoo_answers_qa", trust_remote_code=True)
yahoo_answers_qa = dataset["train"].train_test_split(test_size=0.3)

print(yahoo_answers_qa)

sys.exit(1)

# Preprocess the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    # evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=24,
    per_device_eval_batch_size=24,
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
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

# Fine-tune the model
trainer.train()

# Evaluate the model on the validation dataset
eval_results = trainer.evaluate()

# Print the evaluation results
print(f"Evaluation results: {eval_results}")