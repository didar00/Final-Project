#Packets

import torch
from torch.utils.data.dataset import Dataset

import pandas as pd
from transformers import DataCollatorForLanguageModeling

from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from transformers import RobertaTokenizerFast

from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator

from transformers import get_scheduler
from tqdm.auto import tqdm
import math

#Dataset

class CustomDataset(Dataset):
    def __init__(self, df, tokenizer, MAX_LEN):
        self.examples = []
        
        for example in df.values:
            x = tokenizer.encode_plus(example, max_length = MAX_LEN, truncation=True, padding='max_length')
            self.examples += [x.input_ids]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i])


#Train Model

selfies_path="./data/selfies_subset.txt"
output_dir="./accsaved_model/"

#Tokenizer - Data

tokenizer = RobertaTokenizerFast.from_pretrained("./data/bpe/")

df = pd.read_csv(selfies_path, header=None)
MAX_LEN = 128
train_dataset = CustomDataset(df[0][:100], tokenizer, MAX_LEN) # column name is 0 temp.
eval_dataset = CustomDataset(df[0][100:200], tokenizer, MAX_LEN)

data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

#from transformers import default_data_collator
batch_size = 16
train_batch_size = 16
eval_batch_size = 8
train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator,
)
eval_dataloader = DataLoader(
    eval_dataset, batch_size=batch_size, collate_fn=data_collator
)

#Model

config = RobertaConfig(
    vocab_size=8192,
    max_position_embeddings=514,
    num_attention_heads=2,
    num_hidden_layers=1,
    type_vocab_size=1,
)

model = RobertaForMaskedLM(config=config)

print(torch.cuda.is_available())

optimizer = AdamW(model.parameters(), lr=1e-4)

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)


num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

#Train

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(batch_size)))

    losses = torch.cat(losses)
    losses = losses[: len(eval_dataset)]
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    print(f">>> Epoch {epoch}: Perplexity: {perplexity}")

    # Save and upload
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        