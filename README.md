# Sistema Inteligente de Chamada (UNOESC)

Este projeto é um sistema automatizado de verificação de presença, desenvolvido para o curso de Engenharia da Computação. O sistema utiliza **Visão Computacional** e **Deep Learning** para identificar alunos em tempo real através da webcam, realizando a chamada de forma ágil e segura.

## 🛠️ Arquitetura do Projeto

O sistema é dividido em duas camadas principais:

* **Backend (`backend.py`):** O motor de processamento. Gerencia o carregamento dos modelos de Inteligência Artificial, processa a captura de imagem, detecta rostos e extrai assinaturas biométricas.
* **Frontend (`frontend/frontend.py`):** A interface do usuário. Construída com **Streamlit**, gerencia a interação com a câmera, a exibição do painel de presença e a persistência dos dados na memória da sessão (`st.session_state`).

## 🧠 Tecnologias Utilizadas

* **Linguagem:** Python 3.8+
* **Interface Web:** Streamlit
* **Visão Computacional:** OpenCV (Biblioteca principal)
* **Inteligência Artificial:** * **YuNet:** Modelo de detecção facial (Deep Learning).
    * **SFace:** Modelo de reconhecimento facial e extração de *embeddings* (assinaturas faciais).

## 🚀 Como Executar

### 1. Pré-requisitos
Certifique-se de ter o Python instalado em sua máquina.

### 2. Instalação
No terminal, navegue até a pasta do projeto e instale as dependências necessárias:

```bash
pip install -r requirements.txt
