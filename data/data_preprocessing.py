# Import required libraries and modules
from data_utils import *
import os
import pandas as pd

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


# Read the data from CSV file
data_train = pd.read_csv(data_path)

print('Done!')

# Preprocess the 'discourse_text' column and filter rows with 'Claim' in 'discourse_type'
data_train['discourse_text'] = data_train['discourse_text'].apply(lambda x: clean_sentence(x).lower())
data_train = data_train[data_train['discourse_type'] == paragraph_key]

# Filter out rows with 'discourse_text' length greater than or equal to 70 tokens
discourse_text_length = data_train['discourse_text'].apply(lambda x: len(config['tokenizer'].tokenize(x)))
data_train = data_train[discourse_text_length < 70].reset_index()

# Group by 'id' and aggregate 'discourse_type' and 'discourse_text' into 'label_count' and 'label' respectively
train_df = data_train.groupby(['id']).agg(label_count=('discourse_type', 'count'), label=('discourse_text', '|'.join)).reset_index()

# Read all JSON files and store in train_data_dict
train_data_dict = read_all_json(df=train_df, path=config['train_path'])

# Tokenization and data labelling
train_sentences, train_labels, train_keywords, train_Id_list = labelling(dataset=train_df, data_dict=train_data_dict,config=config)

print("")
print(f" train sentences: {len(train_sentences)}, train label: {len(train_labels)}, train keywords: {len(train_keywords)}, train_id list: {len(train_Id_list)}")

## Create d
unique_df = pd.DataFrame({'id': train_Id_list, 'train_sentences': train_sentences, 'kword': train_keywords, 'label': train_labels})

# Convert 'label' and 'kword' columns to string type
unique_df.label = unique_df.label.astype('str')
unique_df.kword = unique_df.kword.astype('str')

# Add a 'sent_len' column with the length of sentences
unique_df['sent_len'] = unique_df.train_sentences.apply(lambda x: len(x.split(" ")))

# Drop duplicates from the DataFrame
unique_df = unique_df.drop_duplicates()

print(unique_df.shape)
print(unique_df)

unique_df.to_csv(data_transformed_path)

print('The data saved !')


