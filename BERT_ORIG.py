import pandas as pd  # For data manipulation and analysis
import gc  # For garbage collection to manage memory
import re  # For regular expressions
import numpy as np  # For numerical operations and arrays
import tensorflow as tf
from transformers import AutoModel, AutoTokenizer  # Transformers library for natural language processing
from transformers import TextDataset, LineByLineTextDataset, DataCollatorForLanguageModeling, \
pipeline, Trainer, TrainingArguments, DataCollatorWithPadding  # Transformers components for text processing
from transformers import TFAutoModelForSequenceClassification  # Transformer model for sequence classification


import datasets  # Import datasets library
from datasets import Dataset, Image, ClassLabel  # Import custom 'Dataset', 'ClassLabel', and 'Image' classes
from transformers import pipeline  # Transformers library for pipelines
import evaluate

import matplotlib.pyplot as plt  # For data visualization
import itertools  # For working with iterators
from sklearn.metrics import (  # Import various metrics from scikit-learn
    accuracy_score,  # For calculating accuracy
    roc_auc_score,  # For ROC AUC score
    confusion_matrix,  # For confusion matrix
    classification_report,  # For classification report
    f1_score  # For F1 score
)
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers.schedules import PolynomialDecay


from datasets import load_metric

file_name = "news_data.csv"

df = pd.read_csv(file_name)

df = df[['Label', 'News_Headline']]
df = df.rename(columns={'News_Headline': 'text'})
df = df.rename(columns={'Label': 'label'})
df = df[df.label != "full-flop"]
df = df[df.label != "half-flip"]
df = df[df.label != "no-flip"]


df = df.sample(frac = 1.0).reset_index(drop = True)
df_train = df.loc[0: 7999]
df_test = df.loc[8000: ]
train_set = df_train.to_dict(orient= "list")
test_set = df_test.to_dict(orient= "list")

label = list(set(train_set['label']))
print(label)

id2label = {}
label2id = {}
for idx , l in enumerate(label):
    id2label[idx] = l
    label2id[l] = idx


idx = [label2id[l] for l in train_set['label']]
train_set['label'] = idx

idx = [label2id[l] for l in test_set['label']]
test_set['label'] = idx

train_set = Dataset.from_dict(train_set)
test_set = Dataset.from_dict(test_set)


model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
def preprocess(items):
    return tokenizer(items['text'], truncation=True)



tokenzied_train = train_set.map(preprocess, batched = True, load_from_cache_file= False)

tokenzied_test = test_set.map(preprocess, batched = True, load_from_cache_file= False)
print(tokenzied_train)

acc = evaluate.load("accuracy")
data_collator = DataCollatorWithPadding(tokenizer= tokenizer, return_tensors= "tf")



tf_train = tokenzied_train.to_tf_dataset(
    columns = ["attention_mask", "label",  "input_ids"],
    label_cols=["label"],
    shuffle = True,
    collate_fn= data_collator,
    batch_size= 16
)


tf_test = tokenzied_test.to_tf_dataset(
    columns = ["attention_mask", "label", "input_ids"],
    label_cols=["label"],
    shuffle = True,
    collate_fn= data_collator,
    batch_size= 16
)
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return acc.compute(predictions=predictions, references=labels)

num_epochs = 5

model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels = len(label), id2label = id2label, label2id = label2id)
num_train_steps = len(tf_train) * num_epochs

lr_scheduler = PolynomialDecay(
    initial_learning_rate=5e-5, end_learning_rate=0.0, decay_steps=num_train_steps
)
from tensorflow.keras.optimizers import Adam

opt = Adam(learning_rate=lr_scheduler)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
model.fit(tf_train, validation_data=tf_test, epochs=3)


'''
trainer.train()
trainer.evaluate()
trainer.save_model("model_BERT_ONLY")'''


