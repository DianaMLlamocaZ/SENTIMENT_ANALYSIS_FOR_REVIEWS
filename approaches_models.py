import re
import nltk
import sentencepiece as smp
import torch
import numpy as np

from nltk.corpus import stopwords
from utils import load_dl_models_no_attention,load_dl_model_with_Attention

def nltk_download(): #Función que descarga la lista de stop_words --> Evitar la descarga cada vez que se importa el archivo
    nltk.download("stopwords")
    pass



#Clases de preprocesamiento para ambos enfoques:
#ML (Logistic Regression + TF-IDF)
#DL (LSTM, GRU, GRU + Attention Mechanism)

class ApproachML:
    def __init__(self,oracion,log_reg_model,tf_idf_vectorized):
        self.oracion=oracion
        self.log_reg_model=log_reg_model
        self.vectorizer_tf_idf=tf_idf_vectorized
        
        self.contractions_dict={
            "don't": "do not",
            "won't": "will not",
            "can't": "cannot",
            "shan't":"shall not",
            "should've": "should have",
            "it's": "it is","he's":"he is","she's":"she is",
            "i'm": "i am","you're": "you are","we're":"we are","they're":"they are"}
        
        self.clases={
            0:"negativo",
            1:"neutral",
            2:"positivo"
        }
        
        self.stop_words_final=self.stop_word()



    def stop_word(self):
        stop_words=stopwords.words("english")
        not_remove_words=["no", "nor", "not", "few", "more", "most", "too", "very", "will", "might", "need", "would",
                    "did", "do", "had","has", "have", "was","were" "is","are", "shall", "could", "must", "should",
                    "but"]
        
        stop_words_final=[word for word in stop_words if word not in not_remove_words]
        return stop_words_final



    def clean_text(self):
        self.oracion=self.oracion.lower()   #Minúscula
        self.oracion=re.sub('<.*?>',"",self.oracion)    #Remover HTML
        
        self.oracion=" ".join([self.contractions_dict.get(word,word) for word in self.oracion.split()]) #Reemplazo de contracciones de palabras
        self.oracion=re.sub(r"(\w+)n't",r"\1 not",self.oracion) #Detectar patrón word + 'nt que ES contracción directamente. Evitando  "won't" --> wo + 'nt
        
        self.oracion=re.sub(r"(\w+)'s",r"\1",self.oracion)  #Posesivos, elimina el 's
        self.oracion=re.sub(r" \d+", "", self.oracion)  #Elimina números

        self.oracion=re.sub(r"(.)\1{2,}", r"\1\1",self.oracion) #Manejar repeticiones de palabras --> conserva 2 palabras nada más (ver esto)
        


    def predict(self):
        self.clean_text()
        
        text_vectorized=self.vectorizer_tf_idf.transform([self.oracion])    #Vectorización de la opción
        probs_log_reg=self.log_reg_model.predict_proba(text_vectorized) #.item()   #Predicción --> Predice la clase directamente, NO la probabilidad
        clase_pred,prob_pred=np.argmax(probs_log_reg,axis=1)[0],np.max(probs_log_reg,axis=1)[0]
        
        
        return self.clases[clase_pred],prob_pred


#=========#


class ApproachDL:
    def __init__(self,oracion):
        self.oracion=oracion
        self.softmax=torch.nn.Softmax(dim=-1)

        self.contractions_dict={
            "don't": "do not",
            "won't": "will not",
            "can't": "cannot",
            "shan't":"shall not",
            "should've": "should have",
            "it's": "it is","he's":"he is","she's":"she is",
            "i'm": "i am","you're": "you are","we're":"we are","they're":"they are"}
        
        self.clases={
            0:"negativo",
            1:"neutral",
            2:"positivo"
        }
        
        self.stop_words_final=self.stop_word()

        self.tokenizer,self.modelo_attn_no_query,self.lstm_no_attn_ln,self.gru_no_attn_ln=self.load_models()


    def load_models(self):
        #SentencePiece, Attention y LSTM/GRU
        
        #Sentence Piece
        tokenizer_model=smp.SentencePieceProcessor()
        tokenizer_model.load("./saved_models/deep_learning/unigram_model.model")

        #Modelos NO attention mechanism
        lstm_modelo_no_attn_ln,gru_modelo_no_attn_ln=load_dl_models_no_attention()

        #Modelo SÍ attention mechanism
        modelo_attn_no_query=load_dl_model_with_Attention()


        return tokenizer_model,modelo_attn_no_query,lstm_modelo_no_attn_ln,gru_modelo_no_attn_ln


    def stop_word(self):
        stop_words=stopwords.words("english")
        not_remove_words=["no", "nor", "not", "few", "more", "most", "too", "very", "will", "might", "need", "would",
                    "did", "do", "had","has", "have", "was","were" "is","are", "shall", "could", "must", "should",
                    "but"]
        
        stop_words_final=[word for word in stop_words if word not in not_remove_words]
        return stop_words_final
    

    def clean_text(self):
        self.oracion=self.oracion.lower()   #Minúscula
        self.oracion=re.sub('<.*?>',"",self.oracion)    #Remover HTML
        
        self.oracion=" ".join([self.contractions_dict.get(word,word) for word in self.oracion.split()]) #Reemplazo de contracciones de palabras
        self.oracion=re.sub(r"(\w+)n't",r"\1 not",self.oracion) #Detectar patrón word + 'nt que ES contracción directamente. Evitando  "won't" --> wo + 'nt
        
        self.oracion=re.sub(r"(\w+)'s",r"\1",self.oracion)  #Posesivos, elimina el 's
        self.oracion=re.sub(r"([!?,.])\1+",r"\1",self.oracion)  #Evitar signos de puntuación repetidos

        self.oracion=re.sub(r"[^a-zA-Z0-9\s!?,.]","",self.oracion)  #Eliminación de signos de puntuación NO permitidos
        self.oracion=re.sub(r'\d[\d,]*\.?\d*',' NUM ',self.oracion) #"Mantener" los números con "NUM"
        
        self.oracion=re.sub(r"(.)\1{2,}", r"\1\1",self.oracion) #Manejar repeticiones de palabras --> conserva 2 palabras nada más (ver esto)
        self.oracion=" ".join([word for word in self.oracion.split() if word not in self.stop_words_final]) #Remover stop words
        

    def predict(self):
        with torch.no_grad():
            self.clean_text()   #Limpiar la oración
            tokenized_sentence=torch.tensor(self.tokenizer.encode(self.oracion,out_type=int)).unsqueeze(0) #Tokenizar la oración
            len_batch=[tokenized_sentence.shape[1]] #Necesario por los parámetros del forward
            

            self.modelo_attn_no_query.eval()
            preds_dl_attn=self.modelo_attn_no_query(data=tokenized_sentence,length_batch=len_batch)
            clase_pred_attn=torch.argmax(preds_dl_attn,dim=-1).item()
            


            self.lstm_no_attn_ln.eval()
            preds_dl_lstm=self.lstm_no_attn_ln(data=tokenized_sentence,length_batch=len_batch)
            clase_pred_lstm=torch.argmax(preds_dl_lstm,dim=-1).item()
            


            self.gru_no_attn_ln.eval()
            preds_dl_gru=self.gru_no_attn_ln(data=tokenized_sentence,length_batch=len_batch)
            clase_pred_gru=torch.argmax(preds_dl_gru,dim=-1).item()
            


        #Listas que retornan las clases y probabilidades predichas: attn,lstm,gru
        clases_preds=[self.clases[clase_pred_attn],self.clases[clase_pred_lstm],self.clases[clase_pred_gru]]
        probs_preds=[self.softmax(preds_dl_attn).squeeze(0)[clase_pred_attn],self.softmax(preds_dl_lstm).squeeze(0)[clase_pred_lstm],
                     self.softmax(preds_dl_gru).squeeze(0)[clase_pred_gru]]

        return clases_preds,probs_preds
