import tensorflow as tf
import pandas as pd
from transformers import BertTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
from sbert import SBertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from config import Config
from functions import load_data

data = pd.DataFrame(load_data(Config.PROCESSED_DATA_FILE))

labels = list(data.iris_category.unique())


tokenizer = BertTokenizer.from_pretrained(Config.SBERT_MODEL)
model = SBertModel.from_pretrained(Config.SBERT_MODEL, num_labels=len(labels))

(train_data, test_data) = train_test_split(data, train_size=Config.TRAIN_TEST_SPLIT)

train_data_tokenized = tokenizer(list(train_data["abstract_summary"].values), return_tensors="tf", padding=True, truncation=True)

train_dataset = tf.data.Dataset.from_tensor_slices(
    {
        "input_ids": train_data_tokenized["input_ids"], 
        "attention_mask": train_data_tokenized["attention_mask"], 
        "token_type_ids": train_data_tokenized["token_type_ids"], 
        "labels": list(map(lambda x: labels.index(x), train_data["iris_category"].values))
    })

optimizer = tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE)
model.compile(optimizer=optimizer, loss=model.hf_compute_loss)

print("\n\n")
print("Number of training examples: {}".format(len(train_data)))
print("Number of test examples: {}".format(len(test_data)))

model.fit(train_dataset.shuffle(1000).batch(Config.BATCH_SIZE), epochs=Config.TRAIN_EPOCHS_N, batch_size=Config.BATCH_SIZE)

for label_id, label in enumerate(labels):
    model.config.id2label[label_id] = label
    model.config.label2id[label] = label_id

print("Training completed.")

print("Saving the model...", end="")
model.save_pretrained(Config.MODEL_PATH)
print("Done.")

print("Evaluating the model...")

test_data_tokenized = tokenizer(list(test_data["abstract_summary"].values), return_tensors="tf", padding=True, truncation=True)

test_predicted_labels = []

for batch in range(0, len(test_data_tokenized["input_ids"]), Config.BATCH_SIZE):
    logits = model(
        input_ids = test_data_tokenized["input_ids"][batch:batch + Config.BATCH_SIZE],
        attention_mask = test_data_tokenized["attention_mask"][batch:batch + Config.BATCH_SIZE],
        token_type_ids = test_data_tokenized["token_type_ids"][batch:batch + Config.BATCH_SIZE]).logits
    test_predicted_labels.extend(tf.argmax(logits, axis=-1).numpy())

test_target_labels = list(map(lambda x: labels.index(x), test_data.iris_category.values))

report = classification_report(test_target_labels, test_predicted_labels, target_names=labels)

print(report)