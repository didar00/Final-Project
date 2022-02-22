#Get Data
import pandas as pd
import numpy as np
chembl_df= pd.read_csv("demo2.csv", delimiter=";")
selfies_df = chembl_df["Selfies"]
class_df = chembl_df["Class"]

#Get Vectors Hidden and Sequence
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
import torch

config = RobertaConfig.from_pretrained("./molbert_saved_model/model")
config.output_hidden_states = True

tok = RobertaTokenizer.from_pretrained("./robertatokenizer")
model = RobertaModel.from_pretrained("./molbert_saved_model/model", config=config)

#List of Vectors
#Embeds=[]
Hiddens=[]
Seqs=[]
Class=[]
for selfies,classes in zip(selfies_df,class_df):
    try:
        sentence = torch.tensor([tok.encode(selfies)])

        output = model(sentence)  # returns a tuple(sequence_output, pooled_output, hidden_states)
        hidden_states = output[-1]
        hidden_output = hidden_states[0]
        #embedding_output = model.embeddings(sentence) same with hidden states

        sequence_output = output[0]
        #Embeds.append(embedding_output.detach().numpy())
        Hiddens.append(np.mean(hidden_output.detach().numpy()[0], axis=0))#.reshape(-1, 1)) #Tensor that requires grad therefore use detach
        Seqs.append(np.mean(sequence_output.detach().numpy()[0], axis=0))#.reshape(-1, 1)) #(1,token size,768) to (768,)
        Class.append(classes)
    except:
        #Do Nothing if Fails
        continue

"""
Check for requierements
import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf
import datashader.bundling as bd
import matplotlib.pyplot as plt
import colorcet as cl
import matplotlib.colors as mlpcl
import matplotlib.cm as matlpcm
import bokeh.plotting as bpl
import bokeh.transform as btr
import holoviews as hv
import holoviews.operation.datashader as hd
"""
#e=df["Seqs"].to_numpy().reshape(1, -1)
#print(d.shape)
#print(e.shape)
#print(np.array(Seqs))
#print(Seqs_arr.shape)
#print(Seqs_arr[0].shape)

Seqs_arr = np.array(Seqs)
Class_arr = np.array(Class)
Hiddens_arr = np.array(Hiddens)


#Umap Representation
import umap.umap_ as umap
import umap.plot
mapper = umap.UMAP(random_state=123).fit(Hiddens_arr) #the clustering and values will be different, to keep the same graph set the common seed
umap_p = umap.plot.points(mapper, labels=Class_arr)
umap.plot.show(umap_p)

mapper = umap.UMAP(random_state=123).fit(Seqs_arr)
umap_p = umap.plot.points(mapper, labels=Class_arr)
umap.plot.plt.show()