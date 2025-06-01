import streamlit as st
import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import pandas as pd
import io

MODEL_NAME_HUGGINGFACE = "chrnphxbia/djmlp_tiny_analise_sentimento"
MAX_LEN = 128

@st.cache_resource
def carregar_modelo_e_tokenizador(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = TFAutoModelForSequenceClassification.from_pretrained(model_name)

        id_to_label_map = model.config.id2label if hasattr(model.config, 'id2label') and model.config.id2label else {
            0: "positivo",
            1: "neutro",
            2: "negativo"
        }
        if not isinstance(id_to_label_map, dict) or not all(isinstance(k, int) for k in id_to_label_map.keys()):
            st.warning("id2label do modelo não está no formato esperado (dicionário int:str). Usando mapeamento padrão.")
            id_to_label_map = {0: "positivo", 1: "neutro", 2: "negativo"}

        return tokenizer, model, id_to_label_map
    except Exception as e:
        st.error(f"Erro ao carregar o modelo ou tokenizador do Hugging Face: {e}")
        st.error(f"Certifique-se de que o modelo '{model_name}' existe no Hugging Face Hub e que você tem acesso a ele.")
        return None, None, None

def analisar_sentimento(texto, tokenizer, model, id_to_label_map):
    if not isinstance(texto, str) or not texto.strip():
        return None, None

    inputs = tokenizer(
        texto,
        max_length=MAX_LEN,
        truncation=True,
        padding='max_length',
        return_tensors='tf'
    )

    input_dict = {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']}

    try:
        predictions = model.predict(input_dict, verbose=0)
        logits = predictions.logits
        predicted_class_id = np.argmax(logits, axis=1)[0]
        sentiment_label = id_to_label_map.get(predicted_class_id, "desconhecido")
        probabilities = tf.nn.softmax(logits, axis=1)[0].numpy()
        return sentiment_label, probabilities
    except Exception as e:
        st.error(f"Erro durante a predição do modelo para o texto: '{texto[:50]}...'. Erro: {e}")
        return "erro", None


def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')


def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sentimentos')
    processed_data = output.getvalue()
    return processed_data


st.set_page_config(page_title="djlmp_sentiment_analyzer", layout="wide")
st.title("🔍 Análise de Sentimento de Textos em Português")
st.markdown("Utilize o modelo djlmp_sentiment_analyzer para analisar o sentimento de um texto!")

tokenizer, model, id_to_label_map = carregar_modelo_e_tokenizador(MODEL_NAME_HUGGINGFACE)

if tokenizer and model and id_to_label_map:
    tab1, tab2 = st.tabs(["📝 Análise de Texto Único", "📊 Análise de Arquivo (CSV/XLSX)"])

    with tab1:
        st.header("Analisar um único texto")
        texto_usuario = st.text_area("Digite o texto que você gostaria de analisar:", height=150,
                                     placeholder="Ex: Adorei o novo filme, os efeitos especiais são incríveis!",
                                     key="texto_unico")

        if st.button("Analisar Texto 🚀", type="primary", key="analisar_texto_btn"):
            if texto_usuario:
                with st.spinner("Analisando o sentimento... Por favor, aguarde."):
                    sentimento, probabilidades = analisar_sentimento(texto_usuario, tokenizer, model, id_to_label_map)

                if sentimento:
                    st.subheader("Resultado da Análise:")
                    if sentimento.lower() == "positivo":
                        st.success(f"**Sentimento Predito: Positivo** 😊")
                    elif sentimento.lower() == "negativo":
                        st.error(f"**Sentimento Predito: Negativo** 😠")
                    elif sentimento.lower() == "neutro":
                        st.info(f"**Sentimento Predito: Neutro** 😐")
                    else:
                        st.warning(f"Sentimento Predito: {sentimento.capitalize()}")

                    st.markdown("---")
                    st.write("Probabilidades por classe:")
                    if probabilidades is not None:
                        for i, prob in enumerate(probabilidades):
                            label_name = id_to_label_map.get(i, f"Classe {i}")
                            st.markdown(f"*{label_name.capitalize()}*: `{prob:.2%}`")
                else:
                    st.warning("Por favor, insira um texto válido para análise.")
            else:
                st.warning("⚠️ Por favor, digite algum texto na caixa acima antes de analisar.")

    with tab2:
        st.header("Analisar um arquivo de planilha")
        uploaded_file = st.file_uploader("Carregue um arquivo XLSX ou CSV", type=["csv", "xlsx"], key="file_uploader")

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                else:
                    st.error("Formato de arquivo não suportado. Por favor, carregue CSV ou XLSX.")
                    df = None 

                if df is not None:
                    st.write("Pré-visualização dos dados carregados:")
                    st.dataframe(df.head())

                    colunas = df.columns.tolist()
                    coluna_texto_selecionada = st.selectbox("Selecione a coluna que contém os textos para análise:", colunas, key="coluna_selecionada")

                    if st.button("Analisar Arquivo 🚀", type="primary", key="analisar_arquivo_btn"):
                        if coluna_texto_selecionada:
                            sentimentos_previstos = []
                            total_rows = len(df)
                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            with st.spinner(f"Analisando textos da coluna '{coluna_texto_selecionada}'..."):
                                for i, row in enumerate(df.itertuples()):
                                    texto_para_analise = getattr(row, coluna_texto_selecionada)
                                    if pd.isna(texto_para_analise) or not isinstance(texto_para_analise, str):
                                        texto_para_analise = ""

                                    sentimento, _ = analisar_sentimento(str(texto_para_analise), tokenizer, model, id_to_label_map)
                                    sentimentos_previstos.append(sentimento if sentimento else "N/A") 
                                    progress_bar.progress((i + 1) / total_rows)
                                    status_text.text(f"Processando linha {i+1}/{total_rows}")

                            df['sentimento'] = sentimentos_previstos
                            status_text.success(f"Análise concluída! {total_rows} textos processados.")
                            progress_bar.empty() 

                            st.subheader("Resultado da Análise do Arquivo:")
                            st.dataframe(df)

                            st.markdown("---")
                            st.subheader("Download do Arquivo Processado:")

                            csv_data = convert_df_to_csv(df)
                            st.download_button(
                                label="📥 Baixar como CSV",
                                data=csv_data,
                                file_name=f"sentimentos_{uploaded_file.name.split('.')[0]}.csv",
                                mime='text/csv',
                            )

                            excel_data = convert_df_to_excel(df)
                            st.download_button(
                                label="📥 Baixar como XLSX",
                                data=excel_data,
                                file_name=f"sentimentos_{uploaded_file.name.split('.')[0]}.xlsx",
                                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                            )
                        else:
                            st.warning("⚠️ Por favor, selecione uma coluna contendo os textos.")
            except Exception as e:
                st.error(f"Erro ao processar o arquivo: {e}")
                st.error("Verifique se o arquivo está no formato correto e não está corrompido.")

else:
    st.error("A aplicação não pôde ser iniciada pois o modelo não foi carregado do Hugging Face.")
    st.markdown(f"Verifique o console para mensagens de erro e certifique-se de que o nome do modelo (`{MODEL_NAME_HUGGINGFACE}`) está correto e o modelo existe no Hugging Face Hub.")