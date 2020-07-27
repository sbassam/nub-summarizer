import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
import wandb
from torch import cuda
'''
class and functions from 
github.com/abhimishra91/transformers-tutorials
'''

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.summary = self.data.summary
        self.full_text = self.data.full_text

    def __len__(self):
        return len(self.summary)

    def __getitem__(self, index):
        full_text = str(self.full_text[index])
        full_text = ' '.join(full_text.split())

        summary = str(self.summary[index])
        summary = ' '.join(summary.split())

        source = self.tokenizer.batch_encode_plus([full_text], max_length=self.source_len, pad_to_max_length=True,
                                                  return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([summary], max_length=self.summ_len, pad_to_max_length=True,
                                                  return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }


def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    for _, data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype=torch.long)
        mask = data['source_mask'].to(device, dtype=torch.long)

        outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, lm_labels=lm_labels)
        loss = outputs[0]

        if _ % 10 == 0:
            wandb.log({"Training Loss": loss.item()})

        if _ % 500 == 0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



def main(train_dataset, model_output_dir):
    '''function from https://colab.research.google.com/github/abhimishra91/transformers-tutorials/blob/master/transformers_summarization_wandb.ipynb'''
    # WandB – Initialize a new run
    wandb.init(project="transformers_tutorials_summarization")
    device = 'cuda' if cuda.is_available() else 'cpu'

    # WandB – Config is a variable that holds and saves hyperparameters and inputs
    # Defining some key variables that will be used later on in the training
    config = wandb.config  # Initialize config
    config.TRAIN_BATCH_SIZE = 2  # input batch size for training (default: 64)
    config.TRAIN_EPOCHS = 2  # number of epochs to train (default: 10)
    config.LEARNING_RATE = 1e-4  # learning rate (default: 0.01)
    config.SEED = 42  # random seed (default: 42)
    config.MAX_LEN = 512
    config.SUMMARY_LEN = 150

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(config.SEED)  # pytorch random seed
    np.random.seed(config.SEED)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained("t5-base")


    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = CustomDataset(train_dataset.reset_index(drop=True), tokenizer, config.MAX_LEN, config.SUMMARY_LEN)

    # Defining the parameters for creation of dataloaders
    train_params = {
        'batch_size': config.TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
    }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    model = model.to(device)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LEARNING_RATE)
    wandb.watch(model, log="all")

    for epoch in range(config.TRAIN_EPOCHS):
        train(epoch, tokenizer, model, device, training_loader, optimizer)

    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='finetune t-5 base for a given dataset')
    parser.add_argument('--train_dataset', action="store")
    parser.add_argument('--model_output_dir', action="store", default='model/')
    args = parser.parse_args()
    main(args.train_dataset, args.model_output_dir)
