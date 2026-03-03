import cv2

# Coloque aqui o nome da imagem PADRONIZADA (aquela que já foi esticada e endireitada)
# Certifique-se de usar uma imagem ANTES de desenhar os quadrados vermelhos!
caminho_imagem = "teste_2_binarizado.jpg" # Troque para o nome correto do seu arquivo limpo
img = cv2.imread(caminho_imagem)

if img is None:
    print("Erro: Imagem não encontrada.")
else:
    print("INSTRUÇÕES:")
    print("1. Clique e arraste para desenhar um quadrado em volta APENAS das bolinhas do Bloco 1 (Q01 a Q15).")
    print("2. Aperte ENTER ou ESPAÇO para confirmar.")
    print("3. Aperte 'c' para cancelar e tentar de novo.")
    
    # Cria uma janela redimensionável (permite ajustar o tamanho e ver a imagem inteira)
    cv2.namedWindow("Selecione o Bloco", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Selecione o Bloco", 800, 900) # Tamanho inicial ajustado para caber na maioria dos monitores
    
    # Abre a interface interativa
    bbox = cv2.selectROI("Selecione o Bloco", img, fromCenter=False, showCrosshair=False)
    
    # Extrai os valores
    x, y, w, h = bbox
    
    print("\n" + "="*40)
    print("COPIE ESTES VALORES PARA O SEU MAIN.PY:")
    print(f"x_colunas = [{x}, ...]")
    print(f"y_inicio = {y}")
    print(f"largura_bloco = {w}")
    print(f"altura_bloco = {h}")
    print("="*40 + "\n")
    
    cv2.destroyAllWindows()