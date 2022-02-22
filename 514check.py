#Get Data
import pandas as pd
import numpy as np
#chembl_df= pd.read_csv("chembl_29_selfies.csv")
#selfies_df = chembl_df["selfies"]


#Get Vectors Hidden and Sequence
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
import torch

config = RobertaConfig.from_pretrained("./chemberta_saved_model/model")
config.output_hidden_states = True

tok = RobertaTokenizer.from_pretrained("./robertatokenizer")
model = RobertaModel.from_pretrained("./chemberta_saved_model/model", config=config)

#List of Vectors
#Embeds=[]
t=0
all=0
with open('selfies_subset.txt') as selfies_df:
    for selfies in selfies_df:
        try:
            all=all+1
            sentence = torch.tensor([tok.encode(selfies)])
            output = model(sentence)
            print(all)
        except:
            t =t+1
            continue
        
print(t*100/all)
print(all,t)