#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "5"
#os.environ["TOKENIZER_PARALLELISM"] = "false"
#os.environ["WANDB_DISABLED"] = "true"

#Get Data
import pandas as pd
import numpy as np
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
import torch
import pickle

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def embeds(chem_list, num, tok, model):
    
    chembl_df= chem_list
    selfies_df = chembl_df["selfies"]
    ids_df= chembl_df["chembl_id"]

    #List of Vectors
    #Embeds=[]
    Hiddens=[]
    Seqs=[]
    Ids=[]

    for selfies,ids in zip(selfies_df,ids_df):
        try:
            print(ids,len(Ids))
            sentence = torch.tensor([tok.encode(selfies)])

            output = model(sentence)  # returns a tuple(sequence_output, pooled_output, hidden_states)
            hidden_states = output[-1]
            hidden_output = hidden_states[0]
            #embedding_output = model.embeddings(sentence) same with hidden states

            sequence_output = output[0]
            #Embeds.append(embedding_output.detach().numpy())
            Hiddens.append(hidden_output.detach())#.reshape(-1, 1)) #Tensor that requires grad therefore use detach
            Seqs.append(sequence_output.detach())#.reshape(-1, 1)) #(1,token size,768) to (768,)
            #print(np.mean(sequence_output.detach().numpy()[0], axis=0).shape)
            Ids.append(ids)



        except:
            #Do Nothing if Fails
            continue

    Seqs_arr = np.array(Seqs)
    Ids_arr = np.array(Ids)
    Hiddens_arr = np.array(Hiddens)
    #print(Seqs[0],Seqs_arr[0].shape,Hiddens_arr[0].shape)
    
    if(len(Ids) != 0):

        data_s = {'chembl_id':Ids_arr,
                'embeddings':Seqs_arr}
  
        df = pd.DataFrame(data_s)
        df.to_csv('./Sequence_output/sequence_output_embeddings_part_%s.csv' % num, index=False)

        data_h = {'chembl_id':Ids_arr,
                'embeddings':Hiddens_arr}
  
        df_2 = pd.DataFrame(data_h)
        df_2.to_csv('./Hidden_states_output/hidden_states_embeddings_part_%s.csv' % num, index=False)
        
        #with open('./Sequence_output/sequence_output_embeddings_part_%s.pkl' % num, "wb") as fOut:
            #pickle.dump({'chembl_id': Ids, 'sequence_output_embeddings': Seqs}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

        #with open('./Hidden_states_output/hidden_states_embeddings_part_%s.pkl' % num, "wb") as fOut:
            #pickle.dump({'chembl_id': Ids, 'hidden_states_embeddings': Hiddens}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
        return True
    
    else:
        return False

    
    #print(Seqs)
    #print(Hiddens)




if __name__ == "__main__":

    model_name = "./our_90epochs_saved_model"
    config = RobertaConfig.from_pretrained(model_name)
    config.output_hidden_states = True

    tok = RobertaTokenizer.from_pretrained("./robertatokenizer")
    model = RobertaModel.from_pretrained(model_name, config=config)

    chems= pd.read_csv("chembl_29_selfies.csv", delimiter=",")[:5]
    
    chem_chunks = chunks(chems,25000)

    number=1
    for chem in chem_chunks:#pd.read_csv("chembl_29_selfies.csv", delimiter=",", chunksize=25000):

        check = embeds(chem,number,tok,model)
        if (check==True):
            number+=1


    #print(len(seqsofchembs))
    #print(len(hidssofchembs))
    #print(seqsofchembs)

    

#print(Ids[0],chembs_v2[0])
#print(hidssofchembs[0],Hiddens_arr[0])
#print(seqsofchembs[0],Seqs_arr[0])




    
    

   