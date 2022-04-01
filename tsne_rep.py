#Get Data
import pandas as pd
import numpy as np
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
import torch
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

test= pd.read_csv("bbbp.csv_test.txt", sep=",")
train= pd.read_csv("bbbp.csv_train.txt", sep=",")
val= pd.read_csv("bbbp.csv_val.txt", sep=",")

selfies_df = train["selfies"].values.tolist()
class_df = train["p_np"].values.tolist()


selfies_df+= test["selfies"].values.tolist()
selfies_df+= val["selfies"].values.tolist()

class_df += test["p_np"].values.tolist()
class_df += val["p_np"].values.tolist()

#Get Vectors Hidden and Sequence

model_name = "./molbert_bbbp_model/model"
config = RobertaConfig.from_pretrained(model_name)
config.output_hidden_states = True

tok = RobertaTokenizer.from_pretrained("./robertatokenizer")
model = RobertaModel.from_pretrained(model_name, config=config)

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
        print(np.mean(sequence_output.detach().numpy()[0], axis=0).shape)
        Class.append(classes)
    except:
        #Do Nothing if Fails
        continue


Seqs_arr = np.array(Seqs)
Class_arr = np.array(Class)
Hiddens_arr = np.array(Hiddens)

#Hidden Rep
tsne = TSNE(n_components=2, verbose=1, random_state=123,learning_rate='auto',init='random')
z = tsne.fit_transform(Hiddens_arr)
df = pd.DataFrame()
df["y"] = Class_arr
df["comp-1"] = z[:,0]
df["comp-2"] = z[:,1]
sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 2),
                data=df).set(title="Data Hidden Rep T-SNE projection") 
plt.show()

#Seq Rep
tsne = TSNE(n_components=2, verbose=1, random_state=123,learning_rate='auto',init='random')
z = tsne.fit_transform(Seqs_arr)
df = pd.DataFrame()
df["y"] = Class_arr
df["comp-1"] = z[:,0]
df["comp-2"] = z[:,1]
sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 2),
                data=df).set(title="Data Seq Rep T-SNE projection") 
plt.show()