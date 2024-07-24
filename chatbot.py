import json
import random
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Muat model dan tokenizer yang telah dilatih
tokenizer = BertTokenizer.from_pretrained('./model')
model = BertForSequenceClassification.from_pretrained('./model')
print("Model dan tokenizer berhasil dimuat.")

# Muat data dari file JSON
with open('response.json', 'r') as f:
    data = json.load(f)

# Pemetaan dari ID label ke intent
label2id = {label: idx for idx, label in enumerate(list(set([intent['intent'] for intent in data])))}
id2label = {idx: label for label, idx in label2id.items()}

# Prediksi intent
def prediksi_intent(input_user):
    inputs = tokenizer(input_user, return_tensors="pt")
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_label_id = torch.argmax(predictions, dim=-1).item()
    predicted_intent = id2label[predicted_label_id]
    print(f"Input pengguna: '{input_user}'")  # Baris debug
    print(f"Intent yang diprediksi: '{predicted_intent}' dengan kepercayaan {predictions[0][predicted_label_id].item()}")  # Baris debug
    return predicted_intent

# Dapatkan respon berdasarkan intent
def dapatkan_respon(input_user):
    intent = prediksi_intent(input_user)
    for item in data:
        if item['intent'] == intent:
            response = random.choice(item.get('response', ["Maaf, saya tidak mengerti. Bisakah Anda mengulanginya dengan kata lain?"]))
            print(f"Respon yang dipilih: '{response}'")  # Baris debug
            return response
    return "Maaf, saya tidak mengerti. Bisakah Anda mengulanginya dengan kata lain?"
