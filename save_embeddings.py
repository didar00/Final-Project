#Get Data
import pandas as pd
import numpy as np
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
import torch
import pickle

chembl_df= pd.read_csv("chembl_29_selfies.csv", delimiter=",")
selfies_df = chembl_df["selfies"]
ids_df= chembl_df["chembl_id"]


model_name = "./our_90epochs_saved_model"
config = RobertaConfig.from_pretrained(model_name)
config.output_hidden_states = True

tok = RobertaTokenizer.from_pretrained("./robertatokenizer")
model = RobertaModel.from_pretrained(model_name, config=config)

#List of Vectors
#Embeds=[]
Hiddens=[]
Seqs=[]
Ids=[]

for selfies,ids in zip(selfies_df,ids_df):
    try:
        sentence = torch.tensor([tok.encode(selfies)])

        output = model(sentence)  # returns a tuple(sequence_output, pooled_output, hidden_states)
        hidden_states = output[-1]
        hidden_output = hidden_states[0]
        #embedding_output = model.embeddings(sentence) same with hidden states

        sequence_output = output[0]
        #Embeds.append(embedding_output.detach().numpy())
        Hiddens.append(hidden_output)#.reshape(-1, 1)) #Tensor that requires grad therefore use detach
        Seqs.append(sequence_output)#.reshape(-1, 1)) #(1,token size,768) to (768,)
        print(np.mean(sequence_output.detach().numpy()[0], axis=0).shape)
        Ids.append(ids)
    except:
        #Do Nothing if Fails
        continue

Seqs_arr = np.array(Seqs)
Ids_arr = np.array(Ids)
Hiddens_arr = np.array(Hiddens)

with open('sequence_output_embeddings.pkl', "wb") as fOut:
    pickle.dump({'chembl_id': Ids_arr, 'sequence_output_embeddings': Seqs_arr}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

with open('hidden_states_embeddings.pkl', "wb") as fOut:
    pickle.dump({'chembl_id': Ids_arr, 'hidden_states_embeddings': Hiddens_arr}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

#with open('embeddings.pkl', "rb") as fIn:

     #stored_data = pickle.load(fIn)
     #chembs = stored_data['chembs']
     #seqsofchembs = stored_data['Seqs_embeddings']
