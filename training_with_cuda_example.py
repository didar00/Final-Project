from transformers import DistilBertForSequenceClassification, AdamW
from transformers import get_scheduler
from datasets import load_metric
from tqdm import tqdm
import time 
import numpy as np
import torch

#################################

acc = load_metric('accuracy')
precision = load_metric('precision')
recall = load_metric('recall')
f1 = load_metric('f1')

def compute_metrics(predictions, labels):
    acc_result = acc.compute(predictions=predictions, references=labels)
    precision_result = precision.compute(predictions=predictions, references=labels)
    recall_result = recall.compute(predictions=predictions, references=labels)
    f1_result = f1.compute(predictions=predictions, references=labels, labels=np.unique(predictions))

    result = {**acc_result, **precision_result, **recall_result, **f1_result}
    return result
  
#################################

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = DistilBertForSequenceClassification.from_pretrained(modelname)

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)
  
model.to(device)
model.train()

###############################

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

num_epochs = 4
num_training_steps = num_epochs * len(train_loader)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optim,
    num_warmup_steps=100,
    num_training_steps=num_training_steps
)

if lr_schedulerr:
    print('INFO: Initializing learning rate scheduler')
    lr_schedulerr = LRScheduler(optim)
    
################################    

global_preds = []
global_labels = []
train_losses =[]
val_losses = []


for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    with tqdm(train_loader, unit="batch") as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            # calculate loss and train
            optim.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs[0]
            loss.backward()
            optim.step()

            running_loss += loss.item()

            lr_scheduler.step()
          
            # calculate metrics
            _, predicted = torch.max(outputs[1], 1)
            global_preds.extend(predicted.tolist())
            global_labels.extend(labels.tolist())
            metrics = compute_metrics(global_preds, global_labels)
            metrics["loss"] = loss.item()

            # update progress bar
            tepoch.set_postfix(metrics)
            time.sleep(0.1)

        train_loss=running_loss/len(train_loader)
        train_losses.append(train_loss)

    

    # evaluate with eval dataset
    with torch.no_grad():
        running_loss=0
        model.eval()
        val_preds = []
        val_labels = []
        with tqdm(val_loader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description("Evaluate")

                # calculate loss
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                running_loss+=loss.item()

                # calculate metrics
                _, predicted = torch.max(outputs[1], 1)
                val_preds.extend(predicted.tolist())
                val_labels.extend(labels.tolist())
                metrics = compute_metrics(val_preds, val_labels)
                metrics["loss"] = loss.item()

                # update progress bar
                tepoch.set_postfix(metrics)
                time.sleep(0.1)
            
            val_loss=running_loss/len(val_loader)
            val_losses.append(val_loss)

           
    
    if lr_schedulerr:
      lr_schedulerr(val_loss)
      
########################
 

# Test with test datset
with torch.no_grad():
    model.eval()
    test_preds = []
    test_labels = []
    with tqdm(test_loader, unit="batch") as tepoch:
        for batch in tepoch:
            tepoch.set_description("test")
            
            # calculate loss
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
                
            # calculate metrics
            _, predicted = torch.max(outputs[1], 1)
            test_preds.extend(predicted.tolist())
            test_labels.extend(labels.tolist())
            metrics = compute_metrics(test_preds, test_labels)
            metrics["loss"] = loss.item()

            # update progress bar
            tepoch.set_postfix(metrics)
            time.sleep(0.1)
