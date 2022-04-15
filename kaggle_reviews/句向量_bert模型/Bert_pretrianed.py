from pytorch_transformers.optimization import AdamW
import torch
import pandas as pd
from pytorch_transformers import BertTokenizer
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from pytorch_transformers import BertForSequenceClassification, AdamW, BertConfig
import numpy as np

reviews = pd.read_csv("labeledTrainData.tsv", sep = '\t',header = 0, index_col='id')
moods = {0:"negative", 1:"positive"}
print("Total are : %d" % reviews.shape[0])
for label ,mood in moods.items():
    print("{}:{}".format(mood,reviews[reviews.sentiment==label].shape[0]))

sentences = reviews.review.values
labels = reviews.sentiment.values

max_lenth = 0
sentences_lenth = []
for s in sentences:
    sentences_lenth.append(len(s))
    max_lenth = max(max_lenth,len(s))
print("max lenth is ",max_lenth)
plt.plot(sentences_lenth)
plt.ylabel("y")
plt.show()
#choose 6k as max_lenth of sentence

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

input_ids = []
attention_masks = []

for s in sentences:
    encoded_dict = tokenizer.encode(
        s,
        add_special_tokens = True,
        max_lenth = 6000,
        pad_to_max_length = True,
        return_attention_mask = True,
        return_tensors = 'pt',
    )
    
    input_ids.append(encoded_dict['input_ids'])
    #simply differentiates padding from non-padding
    attention_masks.append(encoded_dict['attention_mask'])

# lists to tensors
input_ids = torch.cat(input_ids,dim = 0)
attention_masks = torch.cat(attention_masks,dim = 0)
labels = torch.tensor(labels)

#split to train_dataset and val_dataset
from torch.utils.data import TensorDataset,random_split
dataset = TensorDataset(input_ids)

train_size = int(0.9*len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset,[train_size,val_size])

#dataloaer
batch_size = 32

train_dataloader = DataLoader(
    train_dataset,
    sampler = RandomSampler(train_dataset),
    batch_size = batch_size
)

validation_dataloader = DataLoader(
    val_dataset,
    sampler = RandomSampler(val_dataset),
    batch_size = batch_size
)

#input model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels = 2,
    output_attentions = False,
    output_hidden_states = False,
)
model.cuda()
params = list(model.named_parameters())

                    
#get acc
def flat_accuracy(preds,labels):
    pred_flat = np.argmax(preds,axis = 1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)



