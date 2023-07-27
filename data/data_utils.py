###  Importing 

import os
import pandas as pd
import numpy as np
import json
import re
from nltk.tokenize import sent_tokenize 
from transformers import BertTokenizer, AutoTokenizer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import transformers
from tqdm import tqdm
import glob
from sklearn.model_selection import train_test_split
import datetime 
import warnings
from nltk.tokenize import word_tokenize
warnings.filterwarnings('ignore')

##



def clean_sentence(sentence):
    # Remove non-alphanumeric characters except spaces
    clean_text = re.sub(r"[^a-zA-Z0-9\s]", "", sentence)
    
    # Tokenize the clean text into words
    words = word_tokenize(clean_text)
    
    return ' '.join(words)

def read_all_json(df, path):
    '''
    This function reads all the json input files and 
    return a dictionary containing the id as the key and all the 
    contents of the json as values
    '''
    text_data = {}
    for i, rec_id in tqdm(enumerate(df.id), total = len(df.id)):
        location = f'{path}{rec_id}.txt'

        with open(location, 'r') as f:
            # read and clean the text
            text_data[rec_id] = clean_sentence(' '.join(f.readlines()).replace('\n','')).lower()
        
    print("All files read")
    
    return text_data


def clean_text(txt):
    '''
    This is text cleaning function
    '''
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower())

def data_joining(data_dict_id):
    '''
    This function is to join all the text data from different 
    sections in the json to a single text file. 
    '''
    data_length = len(data_dict_id)

    #     temp = [clean_text(data_dict_id[i]['text']) for i in range(data_length)]
    #temp = [data_dict_id[i]['text'] for i in range(data_length)]
    temp = [data_dict_id[i] for i in range(data_length)]
    temp = '. '.join(temp)
    
    return temp


def make_shorter_sentence(sentence,config):
    '''
    This function is to split the long sentences into chunks of shorter sentences upto the 
    maximum length of words specified in config['MAX_LEN']
    '''
    #sent_tokenized = config['tokenizer'].tokenize(sentence)
    sent_tokenized = sent_tokenize(sentence)

    max_length = config['MAX_LEN']
    overlap = 20 #20
    
    final_sentences = []
    
    for tokenized_sent in sent_tokenized:
        sent_tokenized_clean = clean_text(tokenized_sent)
        sent_tokenized_clean = sent_tokenized_clean.replace('.','').rstrip() 
        
        tok_sent = sent_tokenized_clean.split(" ")
        
        if len(tok_sent)<max_length:
            final_sentences.append(sent_tokenized_clean)
        else :
#             print("Making shorter sentences")
            start = 0
            end = len(tok_sent)
            
            for i in range(start, end, max_length-overlap):
                temp = tok_sent[i: (i + max_length)]
                final_sentences.append(" ".join(i for i in temp))

    return final_sentences


def form_labels(sentence, labels_list,config):
    '''
    This function labels the training data 
    '''
    matched_kwords = []
    matched_token = []
    un_matched_kwords = []
    label = []

    # Since there are many sentences which are more than 512 words,
    # Let's make the max length to be 128 words per sentence.
    tokens = make_shorter_sentence(sentence,config)
    
    for tok in tokens:    
        tok_split = config['tokenizer'].tokenize(tok)
        
        z = np.array(['O'] * len(tok_split)) # Create final label == len(tokens) of each sentence
        matched_keywords = 0 # Initially no kword matched    

        for kword in labels_list:
            if kword in tok: #This is to first check if the keyword is in the text and then go ahead
                kword_split = config['tokenizer'].tokenize(kword)
                for i in range(len(tok_split)):
                    if tok_split[i: (i + len(kword_split))] == kword_split:
                        matched_keywords += 1

                        if (len(kword_split) == 1):
                            z[i] = 'B'
                        else:
                            z[i] = 'B'
                            z[(i+1) : (i+ len(kword_split))]= 'B'

                        if matched_keywords >1:
                            label[-1] = (z.tolist())
                            matched_token[-1] = tok
                            matched_kwords[-1].append(kword)
                        else:
                            label.append(z.tolist())
                            matched_token.append(tok)
                            matched_kwords.append([kword])
                    else:
                        un_matched_kwords.append(kword)
                
    return matched_token, matched_kwords, label, un_matched_kwords




def labelling(dataset, data_dict,config):
    
    '''
    This function is to iterate each of the training data and get it labelled 
    from the form_labels() function.
    '''
    Id_list_ = []
    sentences_ = []
    key_ = []
    labels_ = []
    un_mat = []
    un_matched_reviews = 0


    for i, Id in tqdm(enumerate(dataset.id), total=len(dataset.id)):

        #sentence = data_joining(data_dict[Id])
        sentence = data_dict[Id]
        labels = dataset.label[dataset.id == Id].tolist()[0].split("|")

        s, k, l, un_matched = form_labels(sentence=sentence, labels_list = labels , config=config)

        if len(s) == 0:
            un_matched_reviews += 1
            un_mat.append(un_matched)
        else: 
            sentences_.append(s)
            key_.append(k)
            labels_.append(l)
            Id_list_.append([Id]*len(l))

    print("Total unmatched keywords:", un_matched_reviews)
    sentences = [item for sublist in sentences_ for item in sublist]
    final_labels = [item for sublist in labels_ for item in sublist]
    keywords = [item for sublist in key_ for item in sublist]
    Id_list = [item for sublist in Id_list_ for item in sublist]
    

    return sentences, final_labels, keywords, Id_list