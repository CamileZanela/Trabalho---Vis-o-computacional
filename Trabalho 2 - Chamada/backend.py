import os
import cv2
import numpy as np
import urllib.request

# Limiares ajustados para máxima tolerância em testes locais
LIMIAR_RECONHECIMENTO_DL = 0.75  
LIMIAR_HISTOGRAMA = 0.35         

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")

for pasta in [DATASET_DIR, MODELS_DIR]:
    if not os.path.exists(pasta):
        os.makedirs(pasta)

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
yunet_path = os.path.join(MODELS_DIR, "yunet.onnx")
sface_path = os.path.join(MODELS_DIR, "sface.onnx")

def baixar_modelos_se_nao_existirem():
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    if not os.path.exists(yunet_path):
        try: urllib.request.urlretrieve(YUNET_URL, yunet_path)
        except Exception: urllib.request.urlretrieve("https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx", yunet_path)
    if not os.path.exists(sface_path):
        try: urllib.request.urlretrieve(SFACE_URL, sface_path)
        except Exception: urllib.request.urlretrieve("https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx", sface_path)

def inicializar_ia():
    baixar_modelos_se_nao_existirem()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    detector_dl = cv2.FaceDetectorYN.create(yunet_path, "", (640, 480), 0.8, 0.3, 5000)
    reconhecedor_dl = cv2.FaceRecognizerSF.create(sface_path, "")
    return face_cascade, detector_dl, reconhecedor_dl

face_cascade, detector_dl, reconhecedor_dl = inicializar_ia()

def extrair_caracteristicas_dl(img_bgr, face_haar):
    try:
        h_img, w_img, _ = img_bgr.shape
        detector_dl.setInputSize((w_img, h_img))
        _, faces = detector_dl.detect(img_bgr)
        if faces is not None and len(faces) > 0:
            face_alinhada = reconhecedor_dl.alignCrop(img_bgr, faces[0])
            return reconhecedor_dl.feature(face_alinhada)
        x, y, w, h = face_haar
        fake_face = np.zeros((1, 15), dtype=np.float32)
        fake_face[0, 0:4] = [x, y, w, h]
        face_alinhada = reconhecedor_dl.alignCrop(img_bgr, fake_face)
        return reconhecedor_dl.feature(face_alinhada)
    except Exception: 
        return None

def calcular_histograma(img_bgr, face_rect):
    x, y, w, h = face_rect
    rosto = img_bgr[y:y+h, x:x+w]
    hist = cv2.calcHist([rosto], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def salvar_novo_aluno(nome, foto_upload):
    file_bytes = np.asarray(bytearray(foto_upload.read()), dtype=np.uint8)
    img_np = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img_np is not None:
        nome_limpo = " ".join(nome.strip().split())
        nome_formatado = nome_limpo.title()
        nome_seguro = nome_limpo.lower().replace(" ", "_")
        caminho_salvar = os.path.join(DATASET_DIR, f"{nome_seguro}.jpg")
        
        cv2.imwrite(caminho_salvar, img_np)
        return nome_formatado
    return None

def processar_reconhecimento(frame_bgr):
    """
    Retorna uma tupla: (status_sucesso, mensagem_ou_nome)
    """
    frame_cinza = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces_haarcascade = face_cascade.detectMultiScale(frame_cinza, 1.1, 4, minSize=(40,40))

    if len(faces_haarcascade) == 0:
        return False, "Nenhum rosto foi detectado pelo sensor da câmera. Aproxime-se mais."

    primeira_face = faces_haarcascade[0]
    vetor_webcam = extrair_caracteristicas_dl(frame_bgr, primeira_face)
    hist_webcam = calcular_histograma(frame_bgr, primeira_face)
    
    arquivos_dataset = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not arquivos_dataset:
        return False, "A base de dados (dataset) está vazia."

    melhor_match = None
    melhor_score = 999.0

    for nome_arquivo in arquivos_dataset:
        caminho_foto = os.path.join(DATASET_DIR, nome_arquivo)
        img_aluno = cv2.imread(caminho_foto)
        if img_aluno is not None:
            img_aluno_cinza = cv2.cvtColor(img_aluno, cv2.COLOR_BGR2GRAY)
            faces_aluno = face_cascade.detectMultiScale(img_aluno_cinza, 1.1, 4)
            
            face_aluno_ref = faces_aluno[0] if len(faces_aluno) > 0 else [0, 0, img_aluno.shape[1], img_aluno.shape[0]]
            vetor_aluno = extrair_caracteristicas_dl(img_aluno, face_aluno_ref)
            
            # 1. Reconhecimento Profundo (SFace)
            if vetor_webcam is not None and vetor_aluno is not None:
                dist_cosseno = reconhecedor_dl.match(vetor_webcam, vetor_aluno, 1)
                if dist_cosseno <= LIMIAR_RECONHECIMENTO_DL:
                    if dist_cosseno < melhor_score:
                        melhor_score = dist_cosseno
                        melhor_match = os.path.splitext(nome_arquivo)[0].replace('_', ' ').lower()
            
            # 2. Fallback por Histograma de Cores
            hist_aluno = calcular_histograma(img_aluno, face_aluno_ref)
            score_hist = cv2.compareHist(hist_webcam, hist_aluno, cv2.HISTCMP_CORREL)
            if score_hist >= LIMIAR_HISTOGRAMA and melhor_match is None:
                return True, os.path.splitext(nome_arquivo)[0].replace('_', ' ').lower()

    if melhor_match:
        return True, melhor_match
        
    return False, "Rosto detectado, mas não corresponde a nenhum aluno cadastrado com precisão."