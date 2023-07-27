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
import utils
from utils import *


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




# taking a sample of dataset data validation for model testing

unique_df = pd.read_csv(data_transformed_path)
unique_df = unique_df.sample(int(unique_df.shape[0]*0.1)).reset_index(drop=True)
unique_df.shape

np.random.seed(100)
train_df, valid_df = train_test_split(unique_df, test_size=0.2)

train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)

print(train_df.shape, valid_df.shape)


tags_2_idx = {'O': 0 , 'B': 1, 'P': 2} # 'P' means padding. 
idx_2_tags = {0: 'O' , 1: 'B', 2: 'P'}
def dataset_2_list(df):
    id_list = df.id.values.tolist()
    sentences_list = df.train_sentences.values.tolist()
    keywords_list = df.kword.apply(lambda x : eval(x)).values.tolist()
    
    labels_list = df.label.apply(lambda x : eval(x)).values.tolist()    
    labels_list = [list(map(tags_2_idx.get, lab)) for lab in labels_list]
    
    return id_list, sentences_list, keywords_list, labels_list

final_train_id_list, final_train_sentences, final_train_keywords, final_train_labels = dataset_2_list(df=train_df)
final_valid_id_list, final_valid_sentences, final_valid_keywords, final_valid_labels = dataset_2_list(df=valid_df)

##


train_prod_input = form_input(ID=final_train_id_list, 
                              sentence=final_train_sentences, 
                              kword=final_train_keywords, 
                              label=final_train_labels, 
                              data_type='train')

valid_prod_input = form_input(ID=final_valid_id_list, 
                              sentence=final_valid_sentences, 
                              kword=final_valid_keywords, 
                              label=final_valid_labels, 
                              data_type='valid')

print("from input done !")
train_prod_input_data_loader = DataLoader(train_prod_input, 
                                          batch_size= config['batch_size'], 
                                          shuffle=True)

valid_prod_input_data_loader = DataLoader(valid_prod_input, 
                                          batch_size= config['batch_size'], 
                                          shuffle=True)



print('Start training ... ')
model, val_predictions, val_true_labels,history = train_engine(epoch=config['Epoch'],
                                                       train_data=train_prod_input_data_loader, 
                                                       valid_data=valid_prod_input_data_loader )
