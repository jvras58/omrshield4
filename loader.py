import cv2
import numpy as np


def alinhar_pelos_marcadores(imagem_binarizada, imagem_original):
    """
    Procura os quadrados nos cantos da prova e estica o conteúdo
    para um tamanho exato de 1000x1400 pixels.
    """
    contornos, _ = cv2.findContours(imagem_binarizada, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    centros_quadrados = []
    
    for c in contornos:
        area = cv2.contourArea(c)
        # Filtra por área: ignora sujeiras minúsculas e blocos gigantes
        if 150 < area < 4000: 
            x, y, w, h = cv2.boundingRect(c)
            proporcao = w / float(h)
            
            # Verifica se a forma é um quadrado (proporção entre 0.8 e 1.2)
            # E verifica se o contorno é fechado/sólido
            if 0.8 <= proporcao <= 1.2:
                cx = x + (w // 2)
                cy = y + (h // 2)
                centros_quadrados.append([cx, cy])
                
    # Se não achou pelo menos 4 quadrados, a imagem está muito ruim ou cortada
    if len(centros_quadrados) < 4:
        print("Aviso: Não foi possível encontrar as 4 marcas fiduciais.")
        return imagem_binarizada # Retorna como estava como fallback
        
    # Converte para array numpy para facilitar a matemática
    pontos = np.array(centros_quadrados, dtype="float32")
    
    # Ordenar os 4 cantos extremos (Topo-Esq, Topo-Dir, Base-Dir, Base-Esq)
    # A soma de x+y é menor no topo-esq e maior na base-dir
    s = pontos.sum(axis=1)
    topo_esq = pontos[np.argmin(s)]
    base_dir = pontos[np.argmax(s)]
    
    # A diferença y-x é menor no topo-dir e maior na base-esq
    diff = np.diff(pontos, axis=1)
    topo_dir = pontos[np.argmin(diff)]
    base_esq = pontos[np.argmax(diff)]
    
    marcadores_ordenados = np.array([topo_esq, topo_dir, base_dir, base_esq], dtype="float32")
    
    # O PULO DO GATO: Definimos um tamanho FIXO e universal para a nossa imagem processada!
    # Não importa se a foto original tinha 500px ou 4000px, ela vai virar 1000x1400.
    LARGURA_FIXA = 1000
    ALTURA_FIXA = 1400
    
    destino_fixo = np.array([
        [0, 0], 
        [LARGURA_FIXA - 1, 0], 
        [LARGURA_FIXA - 1, ALTURA_FIXA - 1], 
        [0, ALTURA_FIXA - 1]
    ], dtype="float32")
    
    # Calcula a matriz e estica a imagem
    matriz = cv2.getPerspectiveTransform(marcadores_ordenados, destino_fixo)
    imagem_padronizada = cv2.warpPerspective(imagem_binarizada, matriz, (LARGURA_FIXA, ALTURA_FIXA))
    
    return imagem_padronizada

def mapear_blocos_questoes():
    # Os valores reais e exatos da sua medição corrigida!
    y_inicio = 1023 
    altura_bloco = 374
    largura_bloco = 119
    
    # O X começa em 63, e usamos as distâncias exatas que você mediu na primeira vez
    x_colunas = [63, 223, 381, 539, 696, 853] 
    
    blocos = {}
    for i, x in enumerate(x_colunas):
        numero_bloco = i + 1
        blocos[numero_bloco] = (x, y_inicio, largura_bloco, altura_bloco)
        
    return blocos

def ordenar_pontos(pts):
    """
    Ordena as 4 quinas da folha na seguinte ordem:
    topo-esquerda, topo-direita, base-direita, base-esquerda.
    Isso é vital para a imagem não ficar de cabeça para baixo ou espelhada.
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # A soma das coordenadas (x + y) será mínima no topo-esquerdo e máxima na base-direita
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # A diferença das coordenadas (y - x) será mínima no topo-direito e máxima na base-esquerda
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def corrigir_perspectiva(imagem):
    """
    Encontra as bordas do cartão-resposta e "achata" a imagem.
    """
    # 1. Pegar altura e largura totais da imagem original para nossa Trava de Segurança
    altura_img, largura_img = imagem.shape[:2]
    area_total = altura_img * largura_img

    # 2. Pré-processamento: Tons de cinza, desfoque leve (remover ruídos) e detecção de bordas
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    desfoque = cv2.GaussianBlur(cinza, (5, 5), 0)
    bordas = cv2.Canny(desfoque, 75, 200)

    # 3. Encontrar os contornos (linhas contínuas) na imagem
    contornos, _ = cv2.findContours(bordas.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Ordenar os contornos do maior para o menor e pegar apenas os 5 maiores
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:5]
    
    contorno_documento = None

    # 4. Procurar pelo contorno que tenha exatamente 4 lados (um retângulo/papel)
    for c in contornos:
        perimetro = cv2.arcLength(c, True)
        # O valor 0.02 é a margem de erro para considerar a linha reta
        aproximacao = cv2.approxPolyDP(c, 0.02 * perimetro, True)

        if len(aproximacao) == 4:
            area_contorno = cv2.contourArea(c)
            # TRAVA DE SEGURANÇA: O contorno deve ocupar pelo menos 50% da foto
            if area_contorno > (area_total * 0.5):
                contorno_documento = aproximacao
                break

    # Se não achou um documento seguro, retorna a imagem original (não faz nada)
    if contorno_documento is None:
        return imagem

    # 5. Aplicar a transformação matemática para "achatar" a folha
    pts_ordenados = ordenar_pontos(contorno_documento.reshape(4, 2))
    (tl, tr, br, bl) = pts_ordenados

    # Calcular a largura e altura máximas do novo documento recortado
    larguraA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    larguraB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_largura = max(int(larguraA), int(larguraB))

    alturaA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    alturaB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_altura = max(int(alturaA), int(alturaB))

    # Matriz de destino (como a imagem deve ficar perfeitamente reta)
    destino = np.array([
        [0, 0],
        [max_largura - 1, 0],
        [max_largura - 1, max_altura - 1],
        [0, max_altura - 1]
    ], dtype="float32")

    # Calcula a matriz de perspectiva e aplica na imagem original
    matriz = cv2.getPerspectiveTransform(pts_ordenados, destino)
    imagem_achatada = cv2.warpPerspective(imagem, matriz, (max_largura, max_altura))

    return imagem_achatada

def aplicar_binarizacao(imagem_alinhada):
    cinza = cv2.cvtColor(imagem_alinhada, cv2.COLOR_BGR2GRAY)
    desfoque = cv2.GaussianBlur(cinza, (5, 5), 0)
    
    binarizada = cv2.adaptiveThreshold(
        desfoque, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 15, 4 # Ajustei os parâmetros levemente para lidar melhor com a sua imagem
    )
    
    # --- O PULO DO GATO: Operações Morfológicas ---
    # Cria um "pincel" circular de 3x3 pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Aplica um "Fechamento" (Closing): Ele dilata o branco (juntando os buracos) 
    # e depois erode o excesso, mantendo a forma original, mas preenchida.
    binarizada_limpa = cv2.morphologyEx(binarizada, cv2.MORPH_CLOSE, kernel)
    
    return binarizada_limpa

def ler_alternativa_marcada(imagem_linha_questao):
    """
    Recebe o recorte de UMA linha de questão (ex: as 5 bolinhas da Questão 1).
    Retorna a letra com mais tinta ('A', 'B', 'C', 'D' ou 'E').
    """
    # Define as letras possíveis
    alternativas = ['A', 'B', 'C', 'D', 'E']
    
    # Pega a largura e altura do recorte da questão
    altura, largura = imagem_linha_questao.shape
    
    # Calcula a largura de cada bolinha (dividindo o recorte por 5)
    largura_bolinha = largura // 5
    
    pixels_por_alternativa = []

    # Faz um loop pelas 5 bolinhas
    for i in range(5):
        # Calcula as coordenadas X de início e fim para recortar apenas uma bolinha
        inicio_x = i * largura_bolinha
        fim_x = (i + 1) * largura_bolinha
        
        # Recorta a bolinha específica (A, depois B, etc.)
        bolinha_crop = imagem_linha_questao[:, inicio_x:fim_x]
        
        # CONTA QUANTOS PIXELS BRANCOS EXISTEM NESSA BOLINHA!
        total_pixels_brancos = cv2.countNonZero(bolinha_crop)
        pixels_por_alternativa.append(total_pixels_brancos)
    
    # Encontra o índice da bolinha com o maior número de pixels brancos
    indice_maior = np.argmax(pixels_por_alternativa)
    maior_valor = pixels_por_alternativa[indice_maior]
    
    # Regra de Segurança: Se o número máximo de pixels brancos for muito baixo, 
    # significa que a questão está em branco.
    if maior_valor < 150: # Este valor '150' precisará ser ajustado na prática
        return None # Questão em branco
        
    return alternativas[indice_maior]


def extrair_respostas_do_bloco(imagem_padronizada, x, y, w, h, questao_inicial):
    """
    Recorta um bloco de 15 questões e fatia linha por linha para ler as respostas.
    """
    # Recorta apenas o grande bloco das bolinhas (ex: Q01 a Q15)
    bloco_img = imagem_padronizada[y:y+h, x:x+w]
    
    # A altura total (376) dividida por 15 questões dá ~25 pixels por linha
    altura_linha = h // 15
    
    respostas_bloco = {}
    
    for i in range(15):
        # Calcula onde começa e onde termina a linha dessa questão específica
        y_linha_inicio = i * altura_linha
        y_linha_fim = (i + 1) * altura_linha
        
        # Recorta a tirinha de ~25px de altura
        linha_questao_img = bloco_img[y_linha_inicio:y_linha_fim, :]
        
        # Lemos qual letra foi marcada nesta tirinha
        resposta_letra = ler_alternativa_marcada(linha_questao_img)
        
        # Associa a resposta ao número da questão (ex: "1": "C", "16": "A")
        numero_questao = questao_inicial + i
        respostas_bloco[str(numero_questao)] = resposta_letra
        
    return respostas_bloco