import os
import sys
import cv2
import base64
import numpy as np
import streamlit as st
from datetime import datetime

caminho_atual = os.path.dirname(os.path.abspath(__file__))
caminho_raiz = os.path.dirname(caminho_atual)
if caminho_raiz not in sys.path:
    sys.path.append(caminho_raiz)

import backend

st.set_page_config(
    page_title="Chamada Inteligente - UNOESC",
    page_icon="🦅",
    layout="wide"
)

def carregar_logo_base64(caminho_imagem):
    caminho_raiz_logo = os.path.join(caminho_raiz, caminho_imagem)
    if os.path.exists(caminho_raiz_logo):
        with open(caminho_raiz_logo, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    return ""

logo_b64 = carregar_logo_base64("logo_unoesc.png")

# CSS Premium Dark
st.markdown(f"""
    <style>
    .stApp {{ background-color: #0B0F19 !important; color: #F3F4F6 !important; font-family: 'Inter', sans-serif; }}
    .app-header {{ display: flex; align-items: center; gap: 24px; padding: 20px 0; border-bottom: 2px solid #1F2937; margin-bottom: 30px; }}
    .app-logo {{ height: 75px; object-fit: contain; }}
    .app-title {{ color: #FFFFFF !important; font-size: 2.3rem; font-weight: 700; }}
    .dashboard-card {{ background-color: #111827; border: 1px solid #1F2937; border-radius: 12px; padding: 20px; margin-bottom: 20px; }}
    div[data-testid="stVerticalBlockBorderWrapper"] {{ background-color: #111827 !important; border: 1px solid #1F2937 !important; border-radius: 12px !important; padding: 22px !important; margin-bottom: 20px; }}
    .card-title {{ color: #FFFFFF !important; font-size: 1.15rem; font-weight: 600; }}
    .status-text {{ color: #9CA3AF; font-size: 0.85rem; }}
    .status-dot {{ height: 8px; width: 8px; background-color: #10B981; border-radius: 50%; display: inline-block; margin-right: 6px; }}
    .info-box {{ background-color: rgba(29, 78, 216, 0.15); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 8px; padding: 12px; color: #93C5FD; font-size: 0.85rem; margin-top: 15px; }}
    .student-item {{ display: flex; align-items: center; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #1F2937; }}
    .student-info {{ display: flex; align-items: center; gap: 12px; }}
    .student-avatar {{ width: 36px; height: 36px; border-radius: 50%; background-color: #374151; display: flex; align-items: center; justify-content: center; font-weight: bold; color: #FFFFFF; border: 2px solid #1F2937; }}
    .student-name {{ color: #FFFFFF; font-size: 0.95rem; font-weight: 500; }}
    .student-time {{ color: #6B7280; font-size: 0.8rem; }}
    .badge-present {{ background-color: rgba(16, 185, 129, 0.1); color: #10B981 !important; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; border: 1px solid rgba(16, 185, 129, 0.2); }}
    .badge-absent {{ background-color: rgba(239, 68, 68, 0.1); color: #EF4444 !important; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; border: 1px solid rgba(239, 68, 68, 0.2); }}
    .metric-badge {{ background-color: rgba(16, 185, 129, 0.15); color: #34D399; padding: 2px 8px; border-radius: 4px; font-size: 0.85rem; font-weight: bold; }}
    .stTextInput>div>div>input {{ background-color: #1F2937 !important; border: 1px solid #4B5563 !important; color: #FFFFFF !important; }}
    .stButton>button {{ background-color: #2563EB !important; color: white !important; width: 100%; }}
    .stButton>button:hover {{ background-color: #3B82F6 !important; }}
    #MainMenu, footer, header {{visibility: hidden;}}
    .stDeployButton {{display:none;}}
    </style>
""", unsafe_allow_html=True)

if logo_b64:
    st.markdown(f"<div class='app-header'><img class='app-logo' src='data:image/png;base64,{logo_b64}'><div class='app-title'>Sistema Inteligente de Chamada</div></div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='app-header'><div class='app-title'>🦅 Sistema Inteligente de Chamada • UNOESC</div></div>", unsafe_allow_html=True)

# Memória persistente
if "turma_premium" not in st.session_state:
    st.session_state.turma_premium = {}

if os.path.exists(backend.DATASET_DIR):
    arquivos = [f for f in os.listdir(backend.DATASET_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for arq in arquivos:
        nome_original = os.path.splitext(arq)[0].replace('_', ' ').title()
        if nome_original not in st.session_state.turma_premium:
            st.session_state.turma_premium[nome_original] = {"status": "Ausente", "hora": "--:--"}

col_esq, col_dir = st.columns([1.3, 1], gap="medium")

with col_esq:
    st.markdown("<div class='dashboard-card'><div class='card-title'>Captura de Presença</div><div class='status-text'><span class='status-dot'></span> Sensor Facial Pronto</div></div>", unsafe_allow_html=True)
    
    foto_frame = st.camera_input("Sensor Facial", label_visibility="collapsed")
    
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    recomendar_chamada = st.button("Reconhecer Aluno e Registrar Presença")
    
    st.markdown("<div class='info-box'>ℹ️ <b>Modo Controlado Ativo</b><br>Tire a foto e clique no botão azul acima para processar.</div>", unsafe_allow_html=True)
    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown("<div class='card-title'>Cadastro de fotos</div><div class='status-text'>Adicione fotos dos alunos na base da IA.</div>", unsafe_allow_html=True)
        nome_input = st.text_input("Nome do Aluno:", placeholder="Digite o nome completo...")
        foto_upload = st.file_uploader("Enviar Imagem de Referência:", type=['png', 'jpg', 'jpeg'])
        
        if st.button("Salvar no Banco de Dados"):
            if nome_input and foto_upload:
                nome_formatado = backend.salvar_novo_aluno(nome_input, foto_upload)
                if nome_formatado:
                    st.session_state.turma_premium[nome_formatado] = {"status": "Ausente", "hora": "--:--"}
                    st.success(f"✅ {nome_formatado} cadastrado com sucesso!")
                    st.rerun()
            else:
                st.error("Preencha o nome e carregue uma foto para cadastrar.")

with col_dir:
    total_alunos = len(st.session_state.turma_premium)
    presentes = sum(1 for a in st.session_state.turma_premium.values() if a["status"] == "Presente")
    
    st.markdown(f"<div class='dashboard-card'><div class='card-title'>Presenças - Engenharia da Computação <span class='metric-badge'>{presentes}/{total_alunos}</span></div><div class='status-text'>Lista de presença em tempo real</div><div style='margin-top:15px;'></div>", unsafe_allow_html=True)
    
    if total_alunos == 0:
        st.markdown("<div class='empty-list-text'>Nenhum aluno cadastrado.</div>", unsafe_allow_html=True)
    else:
        for nome, dados in list(st.session_state.turma_premium.items()):
            badge_class = "badge-present" if dados["status"] == "Presente" else "badge-absent"
            texto_rec = f"Reconhecido às {dados['hora']}" if dados["status"] == "Presente" else "Não reconhecido"
            st.markdown(f"<div class='student-item'><div class='student-info'><div class='student-avatar'>{nome[0]}</div><div><div class='student-name'>{nome}</div><div class='student-time'>{texto_rec}</div></div></div><span class='{badge_class}'>{dados['status']}</span></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# LÓGICA REFORMULADA (CORREÇÃO DE RGBA E ALERTA DE ERROS)
if recomendar_chamada:
    if foto_frame and total_alunos > 0:
        bytes_data = foto_frame.getvalue()
        
        # Correção crucial: decodifica a imagem bruta sem forçar BGR direto para não corromper canais alfa/RGBA
        img_np = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_UNCHANGED)
        
        # Se a câmera do navegador enviar formato RGBA (4 canais), converte para BGR (3 canais) para o OpenCV usar
        if img_np is not None and img_np.shape[2] == 4:
            frame_bgr = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
        else:
            frame_bgr = img_np

        if frame_bgr is not None:
            with st.spinner("A processar reconhecimento facial..."):
                status_ia, resultado_ia = backend.processar_reconhecimento(frame_bgr)
            
            if status_ia: # Se encontrou o match com sucesso
                nome_rec_limpo = resultado_ia.strip().lower()
                sucesso_vinculo = False
                
                for nome_painel in list(st.session_state.turma_premium.keys()):
                    nome_painel_limpo = nome_painel.strip().lower()
                    
                    if nome_painel_limpo in nome_rec_limpo or nome_rec_limpo in nome_painel_limpo:
                        st.session_state.turma_premium[nome_painel]["status"] = "Presente"
                        st.session_state.turma_premium[nome_painel]["hora"] = datetime.now().strftime("%H:%M")
                        sucesso_vinculo = True
                
                if sucesso_vinculo:
                    st.toast("Presença confirmada!", icon="✨")
                    st.balloons()
                    st.rerun()
                else:
                    st.error(f"A IA localizou o arquivo '{resultado_ia.title()}', mas esse nome exato não está no painel de chamada.")
            else:
                # Exibe na interface o erro exato retornado pela IA (Ex: Rosto não detectado ou Sem correspondência)
                st.error(f"Erro no Reconhecimento: {resultado_ia}")
    elif total_alunos == 0:
        st.error("Cadastre pelo menos um aluno antes de validar a presença.")
    else:
        st.error("Por favor, capture a foto no visor da câmera antes de clicar.")