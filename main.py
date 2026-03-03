from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
import cv2
from loader import corrigir_perspectiva, aplicar_binarizacao, alinhar_pelos_marcadores, mapear_blocos_questoes, extrair_respostas_do_bloco

app = FastAPI(
    title="API de Leitura de Cartão-Resposta",
    description="API para extrair dados e marcações de cartões-resposta via OCR e OMR.",
    version="1.0.0"
)

def processar_imagem_opencv(img_cv2):
    # 1. Pipeline de Visão Computacional (já fizemos)
    imagem_alinhada = corrigir_perspectiva(img_cv2)
    imagem_binarizada = aplicar_binarizacao(imagem_alinhada)
    imagem_padronizada = alinhar_pelos_marcadores(imagem_binarizada, imagem_alinhada)
    
    # 2. Pega as coordenadas exatas
    coordenadas_blocos = mapear_blocos_questoes()
    
    todas_as_respostas = {}
    
    # Criar uma cópia da imagem padronizada para desenhar (colorida para ver as linhas)
    img_teste = cv2.cvtColor(imagem_padronizada, cv2.COLOR_GRAY2BGR) if len(imagem_padronizada.shape) == 2 else imagem_padronizada.copy()

    # 3. Varre os 6 blocos extraindo as respostas
    for numero_bloco, (x, y, w, h) in coordenadas_blocos.items():
        # Desenha o retângulo do bloco principal (verde)
        cv2.rectangle(img_teste, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Desenha as linhas horizontais de cada questão (vermelho) para melhor visualização
        altura_linha = h // 15
        for i in range(1, 15):
            y_linha = y + (i * altura_linha)
            cv2.line(img_teste, (x, y_linha), (x + w, y_linha), (0, 0, 255), 1)

        # Calcula a primeira questão do bloco (Bloco 1 = 1, Bloco 2 = 16, etc.)
        questao_inicial = ((numero_bloco - 1) * 15) + 1
        
        # Extrai as 15 respostas do bloco atual
        respostas_deste_bloco = extrair_respostas_do_bloco(
            imagem_padronizada, x, y, w, h, questao_inicial
        )
        
        # Junta no dicionário final
        todas_as_respostas.update(respostas_deste_bloco)
        
    # Salva as imagens de debug (fora do loop)
    cv2.imwrite("teste_3_blocos_mapeados.jpg", img_teste)
    cv2.imwrite("teste_1_alinhado.jpg", imagem_alinhada)
    cv2.imwrite("teste_2_binarizado.jpg", imagem_binarizada)

    # 4. Retorna para o Front-End
    return {
        "status": "success",
        "total_questoes_lidas": len(todas_as_respostas),
        "respostas": todas_as_respostas
    }

@app.post("/processar-cartao/")
async def processar_cartao(file: UploadFile = File(...)):
    # 1. Validação básica do tipo de arquivo
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="O arquivo enviado não é uma imagem válida.")

    try:
        # 2. Lendo os bytes da imagem enviada
        contents = await file.read()
        
        # 3. Convertendo os bytes para um array do NumPy
        nparr = np.frombuffer(contents, np.uint8)
        
        # 4. Decodificando o array para o formato de imagem do OpenCV
        img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_cv2 is None:
            raise HTTPException(status_code=400, detail="Não foi possível processar a imagem. Arquivo corrompido?")

        # 5. Chamando a nossa função principal de processamento
        resultado = processar_imagem_opencv(img_cv2)
        
        return resultado

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno ao processar a imagem: {str(e)}")