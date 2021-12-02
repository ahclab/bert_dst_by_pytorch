import torch
import torch.nn as nn
import torch.nn.functional as F
from CustomDataset import CustomDataset
from transformers import BertJapaneseTokenizer
from transformers import AdamW
from tqdm import tqdm
from ExtractionModel import SpanExtraction
from loadData import loadData
from sklearn.model_selection import train_test_split
import wandb
from statistics import mean

PATH_PRETRAINED = PATH
EPOCHS = 15
MAX_LENGTH = 128
BATCH_SIZE = 8

wandb.init(project=project, entity=entity)

criterion=nn.CrossEntropyLoss()

extraction_model = SpanExtraction()
optim = AdamW(extraction_model.parameters(), lr=5e-6)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dst_start, dst_end, isFilled, src = loadData()
dst_start_train, dst_start, dst_end_train, dst_end, isFilled_train, isFilled, src_train, src = train_test_split(dst_start, dst_end, isFilled, src, test_size=0.2)
dst_start_valid, dst_start_test, dst_end_valid, dst_end_test, isFilled_valid, isFilled_test, src_valid, src_test = train_test_split(dst_start, dst_end, isFilled, src, test_size=0.5)


# tokenize
tokenizer = BertJapaneseTokenizer.from_pretrained(PATH_PRETRAINED)

# prepare dataset
inputs_train = tokenizer(src_train, return_tensors='pt', max_length=MAX_LENGTH, truncation=True, padding='max_length')
train_dataset = CustomDataset(inputs_train, isIncluding=isFilled_train, start_idxs=dst_start_train, end_idxs=dst_end_train)

inputs_valid = tokenizer(src_valid, return_tensors='pt', max_length=MAX_LENGTH, truncation=True, padding='max_length')
valid_dataset = CustomDataset(inputs_valid, isIncluding=isFilled_valid, start_idxs=dst_start_valid, end_idxs=dst_end_valid)

inputs_test = tokenizer(src_test, return_tensors='pt', max_length=MAX_LENGTH, truncation=True, padding='max_length')
test_dataset = CustomDataset(inputs_test, isIncluding=isFilled_test, start_idxs=dst_start_test, end_idxs=dst_end_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

extraction_model.initialize()
extraction_model.to(device)

wandb.config = {
  "learning_rate": 1e-6, #下げる 1e-6
  "epochs": EPOCHS,
  "batch_size": BATCH_SIZE, #大きくする 
  "train-size": len(train_dataset),
  "valid-size": len(valid_dataset),
  "test-size": len(test_dataset),
}

# train
num = 0
step_train = 0
step_valid = 0
accumulation_steps = 8

for epoch in range(EPOCHS):
    # setup tqdm
    loss_epoch = []
    train_loop = tqdm(train_loader, leave=True)
    for itr, batch in enumerate(train_loop):
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)
        labels.view(len(input_ids), -1)
        start_idxs = batch["start_idxs"].to(device)
        end_idxs = batch["end_idxs"].to(device)
        # process
        cls_loss, span_logits = extraction_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
        
        loss = cls_loss * 0.8
        
        # span loss
        start_logits = span_logits[0][torch.where(start_idxs>=0)]
        start_idxs = start_idxs[torch.where(start_idxs>=0)]

        if len(start_idxs)>0:
            span_start_loss = criterion(start_logits, start_idxs)

            end_logits = span_logits[1][torch.where(start_idxs>=0)]
            end_idxs = end_idxs[torch.where(end_idxs>=0)]
            span_end_loss = criterion(end_logits, end_idxs)
            span_loss = (span_start_loss + span_end_loss) / 2
            span_loss = (span_loss * labels).mean()
            loss = loss + span_loss * 0.2
        wandb.log({"loss_train": loss.item()})
        wandb.log({"step_train": step_train})
        step_train += 1
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # print relevant info to progress bar
        train_loop.set_description(f'Epoch {epoch}')
        train_loop.set_postfix(loss=loss.item())
        loss_epoch.append(loss.item())

        if (itr + 1)%accumulation_steps==0:
            # update parameters
            optim.step()
            # initialize calculated gradients
            optim.zero_grad()

    wandb.log({"loss-epoch": mean(loss_epoch)})
    
    # validate on each epoch
    valid_loop = tqdm(valid_loader, leave=True)
    loss_epoch = []
    with torch.no_grad():
        for batch in valid_loop:
            # initialize calculated gradients
            optim.zero_grad()

            # pull all tensor batches required for training
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            labels.view(len(input_ids), -1)
            start_idxs = batch["start_idxs"].to(device)
            end_idxs = batch["end_idxs"].to(device)

                # process
            cls_loss, span_logits = extraction_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            
            loss = cls_loss * 0.8
            
            # span loss
            start_logits = span_logits[0][torch.where(start_idxs>=0)]
            start_idxs = start_idxs[torch.where(start_idxs>=0)]

            if len(start_idxs)>0:
                span_start_loss = criterion(start_logits, start_idxs)

                end_logits = span_logits[1][torch.where(start_idxs>=0)]
                end_idxs = end_idxs[torch.where(end_idxs>=0)]
                span_end_loss = criterion(end_logits, end_idxs)
                span_loss = (span_start_loss + span_end_loss) / 2
                span_loss = (span_loss * labels).mean()
                loss = loss + span_loss * 0.2

            wandb.log({"validate-loss": loss.item()})
            wandb.log({"step_valid": step_valid})
            # print relevant info to progress bar
            valid_loop.set_postfix(loss=loss.item())
            loss_epoch.append(loss.item())
            step_valid += 1

    wandb.log({"validate-loss-epoch": mean(loss_epoch)})

    #save model on each loop
    if (epoch+1)%5==1:
        torch.save(extraction_model.state_dict(), "model_{}.pt".format(epoch))
    num += 1

torch.save(extraction_model.state_dict(), "model_last.pt")

# test

