import numpy as np
import pandas as pd 
import torch
import torchtext
from transformers import BertForSequenceClassification , BertTokenizer,AutoModelForTokenClassification
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import transformers
import os

current_directory = os.getcwd()

## Define paths and configurations
# Define paths using os.path.join to create full paths
bert_path = os.path.join(current_directory, "../ChackpointHG") # Full path to the "ChackpointHG" folder
train_path = os.path.join(current_directory, "../datsets/train/") # Full path to the "train" folder inside "datsets"
test_path = os.path.join(current_directory, "") # Empty path, might be updated later
model_path = os.path.join(current_directory, "") # Empty path, might be updated later
model_name = os.path.join(current_directory, "../ChackpointHG") # Full path to the "ChackpointHG" folder
data_path = os.path.join(current_directory, "../datsets/train.csv") # Define the  path to the data file
data_transformed_path =  os.path.join(current_directory, "../datsets/data_preprocessed/data_claim.csv")
paragraph_key = 'Lead'


print("", os.path.join(current_directory, bert_path))  # Print the joined path for debugging

config = {
    'MAX_LEN': 100,
    'tokenizer': AutoTokenizer.from_pretrained(os.path.join(current_directory, bert_path), do_lower_case=True),
    'batch_size': 32,
    'Epoch': 1,
    'train_path': train_path,
    'test_path': test_path, 
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model_path': bert_path,
    'model_name': model_name
}


tags_2_idx = {'O': 0 , 'B': 1, 'P': 2} # 'P' means padding. 
idx_2_tags = {0: 'O' , 1: 'B', 2: 'P'}

class form_input():
    def __init__(self, ID, sentence, kword, label, data_type='test'):
        self.id = ID
        self.sentence = sentence
        self.kword = kword
        self.label = label
        self.max_length = config['MAX_LEN']
        self.tokenizer = config['tokenizer']
        self.data_type = data_type
    
    def __len__(self):
        return len(self.sentence)
    
    def __getitem__(self, item):
        
        #modification here !
        toks = config['tokenizer'].tokenize(self.sentence[item])
        #
        label = self.label[item]

        if len(toks)>self.max_length:
            toks = toks[:self.max_length]
            label = label[:self.max_length]
        
        
        ########################################
        # Forming the inputs
        ids = config['tokenizer'].convert_tokens_to_ids(toks)
        tok_type_id = [0] * len(ids)
        att_mask = [1] * len(ids)
        
        # Padding
        pad_len = self.max_length - len(ids)        
        ids = ids + [2] * pad_len
        tok_type_id = tok_type_id + [0] * pad_len
        att_mask = att_mask + [0] * pad_len
        
        ########################################            
        # Forming the label
        if self.data_type !='test':
            label = label + [2]*pad_len
        else:
            label = 1
            
        
        return {'ids': torch.tensor(ids, dtype = torch.long),
                'tok_type_id': torch.tensor(tok_type_id, dtype = torch.long),
                'att_mask': torch.tensor(att_mask, dtype = torch.long),
                'target': torch.tensor(label, dtype = torch.long)
               }
    



from sklearn.preprocessing import OneHotEncoder 

encoder = OneHotEncoder()
encoder.fit([[0],[1],[2]])

def train_fn(data_loader, model, optimizer):
    '''
    Functiont to train the model
    '''
    loss_fn = nn.CrossEntropyLoss()
    train_loss = 0
    for index, dataset in enumerate(tqdm(data_loader, total = len(data_loader))):
        batch_input_ids = dataset['ids'].to(config['device'], dtype = torch.long)
        batch_att_mask = dataset['att_mask'].to(config['device'], dtype = torch.long)
        batch_tok_type_id = dataset['tok_type_id'].to(config['device'], dtype = torch.long)
        batch_target = dataset['target'].to(config['device'], dtype = torch.long)
        #
        batch_size , seq = batch_input_ids.shape
        #
        batch_target_transormed = torch.tensor(encoder.transform(batch_target.cpu().reshape(-1,1)).toarray()).reshape((batch_size,seq,3)).to(config['device'], dtype = torch.float)
        output = model(batch_input_ids, 
                       token_type_ids=None,
                       attention_mask=batch_att_mask,
                       labels=batch_target)
        
        
        prediction = torch.exp(output.logits)/torch.exp(output.logits).sum(axis = 2).reshape((batch_size,seq,1))
        step_loss = loss_fn(prediction,batch_target_transormed)

        step_loss.backward()
        optimizer.step()        
        train_loss += step_loss/32
        optimizer.zero_grad()
        
    return train_loss.sum()





def eval_fn(data_loader, model):
    '''
    Functiont to evaluate the model on each epoch. 
    We can also use Jaccard metric to see the performance on each epoch.
    '''
    
    model.eval()
    
    eval_loss = 0
    predictions = np.array([], dtype = np.int64).reshape(0, config['MAX_LEN'])
    true_labels = np.array([], dtype = np.int64).reshape(0, config['MAX_LEN'])
    
    with torch.no_grad():
        for index, dataset in enumerate(tqdm(data_loader, total = len(data_loader))):
            batch_input_ids = dataset['ids'].to(config['device'], dtype = torch.long)
            batch_att_mask = dataset['att_mask'].to(config['device'], dtype = torch.long)
            batch_target = dataset['target'].to(config['device'], dtype = torch.long)

            output = model(batch_input_ids, 
                           token_type_ids=None,
                           attention_mask=batch_att_mask,
                           labels=batch_target)

            step_loss = output[0]
            eval_prediction = output[1]

            eval_loss += step_loss
            
            eval_prediction = np.argmax(eval_prediction.detach().to('cpu').numpy(), axis = 2)
            actual = batch_target.to('cpu').numpy()
            
            predictions = np.concatenate((predictions, eval_prediction), axis = 0)
            true_labels = np.concatenate((true_labels, actual), axis = 0)
            
    return eval_loss.sum(), predictions, true_labels




def train_engine(epoch, train_data, valid_data,start_checkpoint = None):
    
    model = transformers.BertForTokenClassification.from_pretrained('bert-base-uncased',  num_labels = len(tags_2_idx))
    
    if start_checkpoint : 
        model.load_state_dict(torch.load(start_checkpoint))
        
    model = nn.DataParallel(model)
    model = model.to(config['device'])
    
    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr= 3e-5)
    best_eval_loss = 1000000

    history = {'train_loss' : [] , 'valid_loss' : []}

    for i in range(epoch):
        train_loss = train_fn(data_loader = train_data, 
                              model=model, 
                              optimizer=optimizer)
        eval_loss, eval_predictions, true_labels = eval_fn(data_loader = valid_data, 
                                                           model=model)
        
        print(f"Epoch {i} , Train loss: {train_loss}, Eval loss: {eval_loss}")
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(eval_loss)

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss           
            
            print("Saving the model")
            torch.save(model.state_dict(), config['model_name'])
            
    return model, eval_predictions, true_labels ,history