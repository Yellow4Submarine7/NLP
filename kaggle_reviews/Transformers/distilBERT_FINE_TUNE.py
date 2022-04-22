from transformers import AutoTokenizer
from datasets import load_datset
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

#Dataprocess

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

IMDBdatas = load_datset("imbd")

def preprocess_function(datas):
    return tokenizer(datas["text"], truncation=True)

tokenized_imdb = IMDBdatas.map(preprocess_function, batched=True)

#collator, 整理
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#Train
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_eval_batch_size=16,
    per_device_train_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset = tokenized_imdb["train"],
    eval_dataset = tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator
)










