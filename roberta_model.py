import torch
from torch.utils.data.dataset import Dataset
import tensorflow as tf


class CustomDataset(Dataset):
    def __init__(self, df, tokenizer, MAX_LEN):
        self.examples = []

        for example in df.values:
            x = tokenizer.encode_plus(example, max_length=MAX_LEN, truncation=True, padding="max_length")
            self.examples += [x.input_ids]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i])


import pandas as pd
from sklearn.model_selection import train_test_split

from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from transformers import RobertaTokenizerFast

from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

import math

from ray import tune
def get_hyper_dict(trial):
    """
    All of these parameters' name must
    be compatible with TrainingArguments'
    hyperparameter names
    """
    # all values have been set to one possible conf.
    # due to the memory error
    # but it STILL gives memory error
    hp = {
        "per_device_train_batch_size": tune.grid_search([8]),
        "per_device_eval_batch_size": tune.grid_search([8]),
        "learning_rate":tune.grid_search([0.001]),
        "weight_decay": tune.grid_search([0.01]),
    }
    return hp

def train_and_save_roberta_model(hyperparameters_dict, selfies_path="./data/selfies_subset.txt", robertatokenizer_path="./data/robertatokenizer/", save_to="./saved_model/"):
    TRAIN_BATCH_SIZE = hyperparameters_dict["TRAIN_BATCH_SIZE"]
    VALID_BATCH_SIZE = hyperparameters_dict["VALID_BATCH_SIZE"]
    TRAIN_EPOCHS = hyperparameters_dict["TRAIN_EPOCHS"]
    LEARNING_RATE = hyperparameters_dict["LEARNING_RATE"]
    WEIGHT_DECAY = hyperparameters_dict["WEIGHT_DECAY"]
    MAX_LEN = hyperparameters_dict["MAX_LEN"]
    

    config = RobertaConfig(
        vocab_size=hyperparameters_dict["VOCAB_SIZE"], 
        max_position_embeddings=hyperparameters_dict["MAX_POSITION_EMBEDDINGS"], 
        num_attention_heads=hyperparameters_dict["NUM_ATTENTION_HEADS"], 
        num_hidden_layers=hyperparameters_dict["NUM_HIDDEN_LAYERS"], 
        type_vocab_size=hyperparameters_dict["TYPE_VOCAB_SIZE"]
        )

    #model = RobertaForMaskedLM(config=config)
    def _model_init():
        return RobertaForMaskedLM(config=config)

    df = pd.read_csv(selfies_path, header=None)

    tokenizer = RobertaTokenizerFast.from_pretrained(robertatokenizer_path)

    # TODO: Test train_test_split with column_name 0
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
    train_dataset = CustomDataset(train_df[0], tokenizer, MAX_LEN)  # column name is 0.
    eval_dataset = CustomDataset(eval_df[0], tokenizer, MAX_LEN)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir=save_to,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        num_train_epochs=TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=VALID_BATCH_SIZE,
        save_steps=8192,
        eval_steps=4096,
        save_total_limit=1
        #fp16=True,
    )

    trainer = Trainer(
        #model=RobertaForMaskedLM(config=config),
        model_init=_model_init,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        #prediction_loss_only=True,
    )

    #from ray import tune
    # the number of gpus
    # that will be used can be
    # either stated in command line
    # or in parameters
    
    tf.compat.v1.disable_eager_execution() # this line should've been enough to solve it, but nah
    trainer.hyperparameter_search(
        hp_space=get_hyper_dict,
        direction="maximize", 
        backend="ray", 
        n_trials=1 # number of trials
    )
    
    
    """
    print(torch.cuda.is_available())
    #torch.cuda.set_device(0)
    print(torch.cuda.current_device())
    print("build trainer with on device:", training_args.device, "with n gpus:", training_args.n_gpu)
    #torch.cuda.set_per_process_memory_fraction(0.3) # limit VRAM usage
    print(torch.cuda.list_gpu_processes())
    """

    """ trainer.train()

    eval_results = trainer.evaluate()
    print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    
    trainer.save_model(save_to) """
