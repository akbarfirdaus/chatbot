import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, load_metric

# Baca data dari file JSON
with open('response.json', 'r') as f:
    data = json.load(f)

patterns = []
intents = []

for intent in data:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        intents.append(intent['intent'])

# Encode label
intent_labels = list(set(intents))
label2id = {label: idx for idx, label in enumerate(intent_labels)}
id2label = {idx: label for label, idx in label2id.items()}
y = [label2id[intent] for intent in intents]

# Konversi data ke format Dataset Hugging Face
dataset = Dataset.from_dict({"text": patterns, "label": y})

# Bagi dataset
train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Load tokenizer dan model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(intent_labels))

# Fungsi preprocessing untuk tokenisasi
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True)

train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# Argumen pelatihan
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=20,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Load metrik akurasi
accuracy_metric = load_metric("accuracy", trust_remote_code=True)

# Fungsi untuk menghitung metrik evaluasi
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.tensor(logits).argmax(dim=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# Buat trainer untuk melatih model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Latih model
trainer.train()

# Evaluasi model
results = trainer.evaluate()
print(f"Akurasi: {results['eval_accuracy'] * 100:.2f}%")

# Simpan model dan tokenizer yang telah dilatih
model.save_pretrained('./model')
tokenizer.save_pretrained('./model')
