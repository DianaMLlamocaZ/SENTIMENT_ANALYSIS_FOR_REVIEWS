import streamlit as st
from approaches_models import ApproachML, ApproachDL
import joblib

#Evitar el scrolling para ver los resultados
st.set_page_config(
    page_title="Predicciones de Texto",
    layout="wide"
)


st.title("Sentiment Analysis")

#Se divide la pantalla en dos columnas para subir los resultados y tener todo a la vista
col_input,col_resultados=st.columns([1,2],gap="large")

with col_input:
    #Input text del usuario
    texto_usuario=st.text_area(
        "Ingresa la review a analizar:", 
        placeholder="Escribe la review",
        height=150
    )

    #Mensaje de advertencia respecto al input text y el dataset utilizado
    st.caption("Nota: Los modelos fueron entrenados con el dataset en inglés Amazon Product Reviews de Kaggle. Entradas en otro idioma pueden afectar la certeza de las predicciones.")
    
    #Botón de enviar debajo del cuadro de texto
    boton_enviar=st.button("Enviar",type="primary")

    st.markdown("<br><br><br><br>",unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("Creado por **Diana**")



#Se evita cargar varias veces la misma función para evitar ralentizar el proceso en la carga del modelo ML, Tf-Idf con bigramas y trigramas
@st.cache_resource
def cargar_modelos_pesados_ml():
    modelo_logreg=joblib.load("./saved_models/machine_learning/logreg_bi_tri_grams_modelo_final.joblib")
    vectorizador_tfidf=joblib.load("./saved_models/machine_learning/tfidf_bi_tri_grams_modelo_final.joblib")
    return modelo_logreg,vectorizador_tfidf



with col_resultados:
    #Procesamiento y cuadros de resultado
    if boton_enviar and texto_usuario.strip():
        with st.spinner("Calculando..."):
            
            #Machine Learning enfoque (Modelo Regresión Logística)
            #Se cargan los modelos directamente aquí
            modelo_logreg,vectorizador_tfidf=cargar_modelos_pesados_ml()


            #Machine Learning enfoque (se instancian, utilizando los modelos ya cargados)
            prepml=ApproachML(texto_usuario,log_reg_model=modelo_logreg,tf_idf_vectorized=vectorizador_tfidf)
            clase_pred_ml,prob_pred_ml=prepml.predict()
            

            #Deep Learning enfoque (LSTM,GRU,GRU+Attention)
            prepdl=ApproachDL(oracion=texto_usuario)
            clases_pred_dl,probs_pred_dl=prepdl.predict()


        #Enfoque Machine Learning
        st.subheader("Enfoque: Machine Learning")


        with st.container(border=True):
            st.markdown("### Clasificador Lineal (Regresión Logística + Tf-Idf 2,3-gram)")
            st.markdown(f"**Predicción:** {clase_pred_ml}")
            st.markdown(f"**Certeza:** {100*prob_pred_ml:.4f}%")


        st.markdown("---") #Línea de división

        
        #Enfoque Deep Learning
        st.subheader("Enfoque: Deep Learning")

        # Dividimos en 3 columnas para tus 3 variantes de DL
        col_dl1,col_dl2,col_dl3=st.columns(3)

        with col_dl1:
            with st.container(border=True):
                st.markdown("### GRU + Attention")
                st.markdown(f"**Predicción:** {clases_pred_dl[0]}")
                st.markdown(f"**Certeza:** {100*probs_pred_dl[0]:.4f}%")

        with col_dl2:
            with st.container(border=True):
                st.markdown("### LSTM")
                st.markdown(f"**Predicción:** {clases_pred_dl[1]}")
                st.markdown(f"**Certeza:** {100*probs_pred_dl[1]:.4f}%")

        with col_dl3:
            with st.container(border=True):
                st.markdown("### GRU")
                st.markdown(f"**Predicción:** {clases_pred_dl[2]}")
                st.markdown(f"**Certeza:** {100*probs_pred_dl[2]:.4f}%")
            
        

    else:
        st.info("Esperando entrada de texto para mostrar los 4 recuadros de resultados.")
