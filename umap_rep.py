#Get Data
import pandas as pd
import numpy as np
chembl_df= pd.read_csv("demo2.csv", delimiter=";")
selfies_df = chembl_df["Selfies"]
class_df = chembl_df["Class"]

#Get Vectors Hidden and Sequence
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
import torch

config = RobertaConfig.from_pretrained("./chemberta_saved_model/model")
config.output_hidden_states = True

tok = RobertaTokenizer.from_pretrained("./robertatokenizer")
model = RobertaModel.from_pretrained("./chemberta_saved_model/model", config=config)

#List of Vectors
#Embeds=[]
Hiddens=[]
Seqs=[]
Class=[]
Selfies=[]
for selfies,classes in zip(selfies_df,class_df):
    try:
        sentence = torch.tensor([tok.encode(selfies)])

        output = model(sentence)  # returns a tuple(sequence_output, pooled_output, hidden_states)
        hidden_states = output[-1]
        hidden_output = hidden_states[0]
        #embedding_output = model.embeddings(sentence) same with hidden states

        sequence_output = output[0]
        #print(embedding_output,"Seq\n",sequence_output)
        #print(embedding_output.shape,sequence_output.shape)
        #Embeds.append(embedding_output.detach().numpy())
        Hiddens.append(np.mean(hidden_output.detach().numpy()[0], axis=0)) #Tensor that requires grad
        Seqs.append(np.mean(sequence_output.detach().numpy()[0], axis=0)) #(1,token size,768) to (768,)
        Class.append(classes)
        Selfies.append(selfies)
    except:
        #Do Nothing if Fails
        continue

print(Hiddens[4][0])
print(Seqs[4].shape) 

#Create Dataframe of Data
import pandas as pd
 
data = {'Sefies':Selfies,
        'Hiddens':Hiddens,
        'Seqs': Seqs,
        'Class': Class}

df = pd.DataFrame(data)

#Umap Representation
import umap
mapper = umap.UMAP().fit(df[["Hiddens", "Class"]])
umap.plot.points(mapper, labels=df["Class"])

mapper = umap.UMAP().fit(df[["Seqs", "Class"]])
umap.plot.points(mapper, labels=df["Class"])

