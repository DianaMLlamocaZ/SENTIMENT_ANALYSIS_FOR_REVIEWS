import joblib
import torch

from architecture_models_dl.gru_no_attn_layer_norm import ModeloGRU
from architecture_models_dl.lstm_no_attn_layer_norm import ModeloLSTM
from architecture_models_dl.gru_attn_no_query_layer_norm import Modelo_Attention

#Funciones auxiliares
def load_ml_model(tf_idf_path,log_reg_path):
    vectorizer_tf_idf=joblib.load(tf_idf_path)
    log_reg=joblib.load(log_reg_path)
    
    return vectorizer_tf_idf,log_reg


def load_dl_models_no_attention():

    #LSTM Modelo
    weights_lstm_no_attn_ln=torch.load("./saved_models/deep_learning/lstm_no_attn_layer_norm.pth",
                                           weights_only=True,map_location=torch.device("cpu"))
        

    lstm_modelo_no_attn_ln=ModeloLSTM(len_vocab=8000,num_clases=3,emb_dim=300,hidden_size=256,num_layers=2,
                         out_l1_dim=128,out_l2_dim=64,bidirectional=False)
    lstm_modelo_no_attn_ln.load_state_dict(weights_lstm_no_attn_ln)


    #GRU Modelo
    weights_gru_no_attn_ln=torch.load("./saved_models/deep_learning/gru_no_attn_layer_norm.pth",
                                           weights_only=True,map_location=torch.device("cpu"))


    gru_modelo_no_attn_ln=ModeloGRU(len_vocab=8000,num_clases=3,emb_dim=300,hidden_size=256,num_layers=2,
                         out_l1_dim=128,out_l2_dim=64,bidirectional=False)
    gru_modelo_no_attn_ln.load_state_dict(weights_gru_no_attn_ln)
    

    return lstm_modelo_no_attn_ln,gru_modelo_no_attn_ln


def load_dl_model_with_Attention():

    #Attention Model (No Query Vector and Layer Norm):
    weights_attn_mod=torch.load("./saved_models/deep_learning/gru_attn_no_query_layer_norm.pth",
                                    weights_only=True,map_location=torch.device("cpu"))
        

    modelo_attn_no_query=Modelo_Attention(len_vocab=8000,num_clases=3,emb_dim=300,hidden_size=256,context_vector_attn_size=128,num_layers=2,
                         out_l1_dim=128,out_l2_dim=64,bidirectional=False)
    modelo_attn_no_query.load_state_dict(weights_attn_mod)
    

    return modelo_attn_no_query
