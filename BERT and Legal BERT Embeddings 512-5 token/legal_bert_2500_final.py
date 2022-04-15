# -*- coding: utf-8 -*-
"""legal_bert_2500_final.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1I4sXOLda-JhJpucEm9HwONk1O-ZxOBwI
"""

import transformers
import numpy as np
import pandas as pd
import torch
import transformers as ppb
import warnings
warnings.filterwarnings('ignore')
from transformers.models.bert.modeling_bert import BertModel as bert
import time
from transformers import AutoTokenizer, AutoModel

dfa = pd.read_csv('merge.csv')
dfa = dfa.drop(['split', 'name' ], axis = 1)
print(len(dfa))
print(dfa.head())

tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")

iterations = int(len(dfa)/100) + 1
c = -1
print(iterations)
for i in range(int(iterations)):
    c += 1
    t0 = time.time()
    print('Iteration no. :' + str(i+1))
    df = dfa[i*100:(i+1)*100]
    index = []
    label = []
    tokenized = df['text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
    max_len = 510
    our_tokens = []
    for j in range(len(tokenized.values)): #sep-cls $ sep-cls
      cur_tokens = tokenized.values[j]
      cur_tokens = cur_tokens[::-1] # reversed
      for k in range(5):
        y = k
        if len(cur_tokens) < y*510:
          break
        index.append(c)
        label.append(df['label'][j])
        tokens_512 = cur_tokens[y*510:(y+1)*510]
        tokens_512 = tokens_512[::-1] # reversed back to original
        if y == 0:
          tokens_512.insert(-1, 102) #sep
        elif y == 4:
          tokens_512.insert(0, 101) #cls
        else:
          tokens_512.insert(-1, 102) #both
          tokens_512.insert(0, 101)
        our_tokens.append(tokens_512)
    max_len = 512
    padded = np.array([i + [0]*(max_len-len(i)) for i in our_tokens])
    attention_mask = np.where(padded != 0, 1, 0)
    attention_mask.shape
    input_ids = torch.tensor(padded)  
    attention_mask = torch.tensor(attention_mask)
    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)
        features = last_hidden_states[0][:,0,:].numpy()

    pd.DataFrame(features).to_csv('features_legal/features__legal_2500tokens' + str(c) + '_.csv',index=False)
    pd.DataFrame(label).to_csv('labels_legal/labels__legal_2500tokens' + str(c) + '_.csv',index=False)
    pd.DataFrame(index).to_csv('id_legal/id__legal_2500tokens' + str(c) + '_.csv',index=False)
    t1 = time.time()
    print(str((t1-t0)/60) + ' mins')

