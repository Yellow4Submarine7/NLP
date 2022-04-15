import os
import random
from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME
from torch._C import device
from Bert_pretrianed import flat_accuracy, model
from Bert_pretrianed import train_dataloader
import time
import datetime
from pytorch_transformers.optimization import AdamW
import torch
import pandas as pd
from pytorch_transformers import BertTokenizer
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from pytorch_transformers import BertForSequenceClassification, AdamW, BertConfig
from pytorch_transformers import get_linear_schedule_with_warmup
import numpy as np

#set hyper parameters
optimizer = AdamW(model.parameters(),lr = 0.005 ,eps = 0.001)
epochs = 4
total_steps = len(train_dataloader) * epochs
scheuler = get_linear_schedule_with_warmup(optimizer,\
                                            num_warmup_steps = 0,\
                                            num_training_steps = total_steps)

output_dir = "./binary_models/"
output_model_file = os.path.join(output_dir,WEIGHTS_NAME)
output_config_file = os.path.join(output_dir,CONFIG_NAME)

#set random seed
seed_val = 520
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

#loss acc time
training_datas = []

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds = elapsed_rounded))

total_t0 = time.time()
best_val_accuracy = 0

for epoch_i in range(0,epochs):
    print('Epoch {:} / {:}'.format(epoch_i + 1,epochs))
    t0 = time.time()
    total_train_loss = 0
    total_train_accuracy = 0
    model.train()

    for step,batch in enumerate(train_dataloader):
        if step % 40 == 40 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print("Batch {:>5} of {:>5,}.  Elapsed: {:}.".format(step,len(train_dataloader),elapsed))
       
        batch_input_ids = batch[0].to(device)
        batch_input_mask = batch[1].to(device) 
        batch_labels = batch[2].to(device)

        model.zero_grad()

        loss,logits = model(batch_input_ids,
                            token_type_ids = None,
                            attention_mask = batch_input_mask,
                            labels = batch_labels)
        
        total_train_loss += loss.item()
        loss.bachward()
        #clip_grad_norm
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)

        #refresh optimizer
        optimizer.step()
        #refresh learning rate
        scheuler.step()

        logit = logits.detach().cpu().numpy()
        label_id = batch_labels.to('cpu').numpy()

        total_train_accuracy += flat_accuracy(logit, label_id)

    avg_train_loss = total_train_loss / len(train_dataloader)
    training_time = format_time(time.time() - t0)
    avg_train_accuracy = total_train_accuracy / len(train_dataloader)
    print("avg_train_accuracy:{0:.2f}".format(avg_train_accuracy))
    print("avg_train_loss:{0:.2f}".format(avg_train_loss))
    print("training_time:{:}".format(training_time))




