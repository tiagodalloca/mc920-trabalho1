import sys
import math
from scipy import misc
from scipy import ndimage
import cv2
import numpy as np
import matplotlib.pyplot as plt
import ntpath

def filtrar(imagem, filtro):
    (altura_imagem, largura_imagem) = imagem.shape[:2]
    (_, largura_filtro) = filtro.shape[:2]
    borda = (largura_filtro - 1) // 2
    imagem_com_borda = cv2.copyMakeBorder(imagem, borda, borda, borda, borda, cv2.BORDER_REPLICATE)
    imagem_result = np.zeros((altura_imagem, largura_imagem), dtype="float32")
    for y in np.arange(borda, altura_imagem + borda):
        for x in np.arange(borda, largura_imagem + borda):
            pedaco_imagem = imagem_com_borda[y - borda:y + borda + 1, x - borda:x + borda + 1]
            pixel_result = (pedaco_imagem * filtro).sum()
            imagem_result[y - borda, x - borda] = pixel_result
    return imagem_result

def filtrar_duplo(imagem, filtro1, filtro2):
    (altura_imagem, largura_imagem) = imagem.shape[:2]
    (_, largura_filtro) = filtro1.shape[:2]
    borda = (largura_filtro - 1) // 2
    imagem_com_borda = cv2.copyMakeBorder(imagem, borda, borda, borda, borda, cv2.BORDER_REPLICATE)
    imagem_result = np.zeros((altura_imagem, largura_imagem), dtype="float32")
    for y in np.arange(borda, altura_imagem + borda):
        for x in np.arange(borda, largura_imagem + borda):
            pedaco_imagem = imagem_com_borda[y - borda:y + borda + 1, x - borda:x + borda + 1]
            pixel_result1 = (pedaco_imagem * filtro1).sum()
            pixel_result2 = (pedaco_imagem * filtro2).sum()
            imagem_result[y - borda, x - borda] = math.sqrt(pixel_result1*pixel_result1 + pixel_result2*pixel_result2)
    return imagem_result



filtros = {}

filtros['h1'] = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
filtros['h2'] = np.array([[-1, -2, 1], [0, 0, 0], [1, 2, 1]])
filtros['h3'] = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
filtros['h4'] = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])*1/9.0
filtros['h5'] = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]])
filtros['h6'] = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])
filtros['h7'] = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
filtros['h8'] = np.array([[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]])
filtros['h9'] = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]])*1/256.0

for image_path in sys.argv[1:]:
    image_name = ntpath.basename(image_path)
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for filtro_nome, filtro in filtros.items():
        print(filtro_nome, filtro)
        img_filtrada = filtrar(img_gray, filtro)
        cv2.imwrite('imagens_filtradas_parte2/' + filtro_nome + '_' + image_name, img_filtrada)

    filtro_h1 = filtros['h1']
    filtro_h2 = filtros['h2']
    img_filtrada = filtrar_duplo(img_gray, filtro_h1, filtro_h2)
    cv2.imwrite('imagens_filtradas_parte2/h1_h2_' + image_name, img_filtrada)

