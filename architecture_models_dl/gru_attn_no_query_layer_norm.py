import torch
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence


class Attention(torch.nn.Module):
    def __init__(self,bidirectional,hidden_size_model,context_vector_size):
        super().__init__()
        self.bidirectional=bidirectional
        
        self.U=torch.nn.Linear(in_features=hidden_size_model*bidirectional,out_features=context_vector_size)
        self.V=torch.nn.Linear(in_features=context_vector_size,out_features=1)

         
        self.tanh=torch.nn.Tanh()
        self.softmax=torch.nn.Softmax(dim=1)


    def forward(self,query,keys,batch_lengths):
        
        #Key vectors: [batch_size,seq_length,D*h_out], D:bidirectional or not
        #--> will contain a concatenation of the forward and reverse hidden states at each time step in the sequence (PyTorch Source)
        keys_vector=self.U(keys)    #--> AT EATCH time step, forward and reverse hidden states
        

        #Tanh activation function para los key vectors
        tanh=self.tanh(keys_vector) #Cuando NO hay query,  #[batch_size,seq_length,context_vector_size]
        


        #Weights attention scores de cada HIDDEN STATE
        v=self.V(tanh).squeeze(2)                                #[batch_size,sequence_length,1] --> [batch_size,sequence_length]
        weights_attention=self.softmax(v).unsqueeze(1)           #[batch_size,1,sequence_length]
        

       
        context_vector=torch.bmm(weights_attention,keys)    #--> weights_attention [batch_size,1,sequence_length]*keys [batch_size,sequence_length,D*Hout]
                                                                            #--> context_vector: [batch_size,1,D*Hout]
        context_vector_final=context_vector.squeeze(1)   #--> context_vector_final: [batch_size,D*Hout]
        
        
        return context_vector_final
    


class Modelo_Attention(torch.nn.Module):
    def __init__(self,len_vocab,num_clases,emb_dim,hidden_size,context_vector_attn_size,num_layers,out_l1_dim,out_l2_dim,bidirectional:bool):
        super().__init__()
        self.num_emb=len_vocab
        self.bid=2 if bidirectional else 1
        self.embedding_layer=torch.nn.Embedding(num_embeddings=self.num_emb,embedding_dim=emb_dim)
        
        self.gru=torch.nn.GRU(input_size=emb_dim,hidden_size=hidden_size,num_layers=num_layers,batch_first=True,bidirectional=bidirectional)
        
        self.ln1=torch.nn.Linear(in_features=hidden_size*self.bid,out_features=out_l1_dim)
        self.ln2=torch.nn.Linear(in_features=out_l1_dim,out_features=out_l2_dim)
        self.ln3=torch.nn.Linear(in_features=out_l2_dim,out_features=num_clases)
        self.relu=torch.nn.ReLU()
        
        self.dropout=torch.nn.Dropout(p=0.1)
        self.layer_norm1=torch.nn.LayerNorm(out_l1_dim)
        self.layer_norm2=torch.nn.LayerNorm(out_l2_dim)
        
        self.attention=Attention(bidirectional=self.bid,hidden_size_model=hidden_size,context_vector_size=context_vector_attn_size)

    
    def forward(self,data,length_batch):
        x=self.embedding_layer(data) #[num_batch,seq_length,embedding_dim]

        padded_sequence=pack_padded_sequence(input=x,lengths=length_batch,batch_first=True,enforce_sorted=False) ##-->> OJITO
        
        
        #GRU                                                        
        hidden_states_no_padding,final_hidden_state_real=self.gru(padded_sequence) #[num_batch,time_steps,emb_dim],[D*num_layers,batch_size,hidden_size], D:2 if bidirectional, else 1
        
        
        #Pad Packed Sequence para los hidden states y el attention mechanism
        hidden_states_padding,_=pad_packed_sequence(sequence=hidden_states_no_padding,batch_first=True)
        
        
        #Attention
        batch_context_vectors=self.attention(query=final_hidden_state_real,keys=hidden_states_padding,batch_lengths=length_batch)


        #Linear layers
        x=self.relu(self.dropout(self.layer_norm1(self.ln1(batch_context_vectors))))
        x=self.relu(self.dropout(self.layer_norm2(self.ln2(x))))
        x=self.ln3(x)
        
        return x
