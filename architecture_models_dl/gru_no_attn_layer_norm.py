import torch
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

class ModeloGRU(torch.nn.Module):
    def __init__(self,len_vocab,num_clases,emb_dim,hidden_size,num_layers,out_l1_dim,out_l2_dim,bidirectional:bool):
        super().__init__()
        self.num_emb=len_vocab
        self.bid=2 if bidirectional else 1
        
        self.embedding_layer=torch.nn.Embedding(num_embeddings=self.num_emb,embedding_dim=emb_dim,padding_idx=0)
        
        self.gru=torch.nn.GRU(input_size=emb_dim,hidden_size=hidden_size,num_layers=num_layers,batch_first=True,bidirectional=bidirectional)
        
        self.ln1=torch.nn.Linear(in_features=hidden_size*self.bid,out_features=out_l1_dim) #--> concatenar en vez de multiplicar las múltiples salidas
        self.ln2=torch.nn.Linear(in_features=out_l1_dim,out_features=out_l2_dim)
        self.ln3=torch.nn.Linear(in_features=out_l2_dim,out_features=num_clases) #num_clases=5
        self.relu=torch.nn.ReLU()
        
        self.dropout=torch.nn.Dropout(p=0.1)
        self.layer_norm_1=torch.nn.LayerNorm(out_l1_dim)
        self.layer_norm_2=torch.nn.LayerNorm(out_l2_dim)


    def forward(self,data,length_batch):
        x=self.embedding_layer(data) #[num_batch,seq_length,embedding_dim]
        
        padded_sequence=pack_padded_sequence(input=x,lengths=length_batch,batch_first=True,enforce_sorted=False) ##-->> OJITO
        
        #GRU                                                        
        hidden_states_no_padding,final_hidden_state_real=self.gru(padded_sequence) #[num_batch,time_steps,emb_dim],[D*num_layers,batch_size,hidden_size], D:2 if bidirectional, else 1

        hidden_states_padding,_=pad_packed_sequence(sequence=hidden_states_no_padding,batch_first=True)  #--> [N,L,D*Hout], [:,:,:mitad] forward, [:,:,mitad:] backward en 't'


        x_rsh,indexes=torch.max(hidden_states_padding,dim=1) #[batch_size,hidden_state]
        
       
        x=self.relu(self.dropout(self.layer_norm_1(self.ln1(x_rsh))))
        x=self.relu(self.dropout(self.layer_norm_2(self.ln2(x))))
        x=self.ln3(x)
        
        return x
