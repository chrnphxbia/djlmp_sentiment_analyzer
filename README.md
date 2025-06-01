# djlmp_sentiment_analyzer

Este projeto realiza análise de sentimentos em textos utilizando um modelo de aprendizado de máquina construído a partir do modelo pré-treinado [BERTimbau](https://huggingface.co/neuralmind/bert-base-portuguese-cased). Ele abrange desde o treinamento do modelo até a hospedagem de uma aplicação *Streamlit* que consome o modelo desenvolvido.

## 📚 Visão Geral

O projeto é dividido em duas partes principais:

1. **Treinamento do Modelo**: Realizado em um *notebook* no *Google Colab*, onde o modelo é treinado para classificar sentimentos em textos.
2. **Aplicação Web**: Desenvolvida com *Streamlit*, permitindo que usuários insiram textos e recebam a análise de sentimento correspondente.

## 🧪 Treinamento do Modelo

O treinamento do modelo foi realizado no seguinte *notebook* do *Google Colab*:

🔗 [Notebook de Treinamento](https://colab.research.google.com/drive/1V4QV0cRvzC_kD7VP3-IERHT2zWapuPcL#scrollTo=3WHgFkjcXKW0)

Neste notebook, são abordadas etapas como:

- Pré-processamento de dados
- Treinamento e avaliação do modelo
- Upload para *Hugging Face Hub* e consumo do modelo

> **Nota**: O [*notebook*](https://github.com/chrnphxbia/djlmp_sentiment_analyzer/blob/main/djlmp_sentiment_analyzer_clean.ipynb) disponível neste repositório não apresenta as saídas da execução das células, pois, por algum motivo, o *notebook* apresentava erro e não era exibido se as saídas não fossem limpas. Portanto, para visualizar o resultado da execução das células, consulte o *notebook* no *link* acima.

## 🌐 Aplicação Streamlit

A aplicação permite que usuários insiram textos e obtenham a análise de sentimento correspondente:

🔗 [Aplicação Streamlit](https://djlmpsentimentanalyzer.streamlit.app/)

## 🎬 Vídeo de Apresentação do Projeto

Confira o vídeo de apresentação do projeto no *YouTube*!

🔗 [Vídeo de Apresentação do Projeto](https://youtu.be/kYSd-RsIzTg)

## 🗂️ Estrutura do Repositório

O repositório contém os seguintes arquivos e diretórios principais:

- `app.py`: Código-fonte da aplicação *Streamlit*.
- `djlmp_sentiment_analyzer_clean.ipynb`: *Notebook* com o processo de treinamento do modelo.
- `djlmp_sentiment_analyzer.pdf`: Artigo do projeto no formato `.pdf`.
- `requirements.txt`: Lista de dependências necessárias para executar o projeto.
- `teste.csv`: Conjunto de dados de exemplo para testes.

> **Nota**: O diretório `.devcontainer` é específico para configurações de ambiente de desenvolvimento. Ignore-o.

## 💜 Desenvolvedores
- [DiegoAluizio](https://github.com/DiegoAluizio)
- [Jonatas-G-Oliveira](https://github.com/Jonatas-G-Oliveira)
- [lihviaa](https://github.com/lihviaa)
- [Marina-Martin](https://github.com/Marina-Martin)
- [chrnphxbia](https://github.com/chrnphxbia)
