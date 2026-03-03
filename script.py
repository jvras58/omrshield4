import cv2

# Coloque a sua imagem PADRONIZADA (limpa, sem os quadrados desenhados)
caminho_imagem = "teste_2_binarizado.jpg" 
img = cv2.imread(caminho_imagem)

if img is None:
    print("Erro: Imagem não encontrada.")
else:
    # 1. Definimos uma altura segura para caber no seu monitor (800 pixels)
    altura_tela = 800
    altura_original, largura_original = img.shape[:2]
    
    # 2. Calculamos o fator de escala exato para não distorcer as coordenadas
    escala = altura_original / altura_tela
    largura_tela = int(largura_original / escala)
    
    # 3. Redimensionamos a imagem APENAS para exibir na sua tela
    img_exibicao = cv2.resize(img, (largura_tela, altura_tela))
    
    print("INSTRUÇÕES:")
    print("1. Desenhe o quadrado pegando APENAS as bolinhas do Bloco 1 (Q01 a Q15).")
    print("2. Aperte ENTER ou ESPAÇO para confirmar.")
    
    # 4. Abre a interface com a imagem que cabe na tela
    bbox = cv2.selectROI("Selecione o Bloco", img_exibicao, fromCenter=False, showCrosshair=False)
    
    x_tela, y_tela, w_tela, h_tela = bbox
    
    # 5. O PULO DO GATO: Multiplicamos o que você desenhou pela escala 
    # para achar o pixel real na imagem de 1400px!
    x_real = int(x_tela * escala)
    y_real = int(y_tela * escala)
    w_real = int(w_tela * escala)
    h_real = int(h_tela * escala)
    
    print("\n" + "="*40)
    print("VALORES REAIS E EXATOS PARA O MAIN.PY:")
    print(f"x_colunas = [{x_real}, ...]")
    print(f"y_inicio = {y_real}")
    print(f"largura_bloco = {w_real}")
    print(f"altura_bloco = {h_real}")
    print("="*40 + "\n")
    
    cv2.destroyAllWindows()