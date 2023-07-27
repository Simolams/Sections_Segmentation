import os
from transformers import AutoTokenizer
import torch

# Get the current working directory
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
