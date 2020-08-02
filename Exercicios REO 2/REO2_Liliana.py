########################################################################################################################
# DATA: 17/07/2020
# DISCIPLINA: VISÃO COMPUTACIONAL NO MELHORAMENTO DE PLANTAS
# ALUNA: LILIANA ROCIVALDA GOMES LEITAO
# GITHUB: lilianaleitao
# PROFESSOR: VINÍCIUS QUINTÃO CARNEIRO
# GITHUB: vqcarneiro
########################################################################################################################
# EXERCÍCIO 01:
########################################################################################################################
# Importar pacotes
import cv2 # Importa o pacote opencv
import numpy as np # Importa o pacote numpy
from matplotlib import pyplot as plt # Importa o pacote matplotlib
########################################################################################################################
# Leitura da imagem
arquivo = "abobrinhas.jpg" # Nome do arquivo a ser utilizado na análise
imagem = cv2.imread(arquivo,1) # Carrega imagem (1 - Colorida (BGR))
imagem = cv2.cvtColor(imagem,cv2.COLOR_BGR2RGB)
########################################################################################################################
# a) Apresente a imagem e as informações de número de linhas e colunas; número de canais e número total de pixels;
print('1 a)')

# Apresentar imagens
# Imagem Em RGB
plt.figure('1a')
plt.imshow(imagem)
plt.title("Abobrinhas")
plt.show()

# Informações
print('INFORMAÇÕES')
print('--------------------------------------------------------------------------------------------------------------')
lin, col, canais = np.shape(imagem)
print('Número de linhas: ' + str(lin))
print('Número de colunas: ' + str(col))
print('Número de canais: ' + str(canais))
print('Número total de pixel: ' + str(lin) +' x '+ str(col))
print('--------------------------------------------------------------------------------------------------------------')
##################################################################################################################

# b) Faça um recorte da imagem para obter somente a área de interesse.Utilize esta imagem para a solução das próximas
# alternativas.
print('1 b) Recorte da imagem')

#Recortar e salvar imagem
imagem = cv2.imread('abobrinhas.jpg')
recorte = imagem[100:, 0:650]
cv2.imwrite("recorte.jpg", recorte)

#Converter imagem em RGB e plotar figura
img_rc = 'recorte.jpg'
img_bgr = cv2.imread(img_rc,1)
img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
plt.figure('1b')
plt.imshow(img_rgb)
plt.title("Recorte")
plt.show()

# Informações
lin2, col2, canais2 = np.shape(img_rgb)
print('Número de linhas: ' + str(lin2))
print('Número de colunas: ' + str(col2))
print('Número de canais: ' + str(canais2))
print('Número total de pixel: ' + str(lin2) +' x '+ str(col2))
print('--------------------------------------------------------------------------------------------------------------')
########################################################################################################################

# c) Converta a imagem colorida para uma de escala de cinza (intensidade) e a apresente utilizando os mapas de cores
# “Escala de Cinza” e “JET”;
print('1 c) Converter imagem colorida em escala cinza')

# Converter imagem colorida em escala cinza
img_rc = 'recorte.jpg'
img_bgr = cv2.imread(img_rc,1)
img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
img_cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Mapas de cores
plt.figure('1c')
plt.subplot(1,2,1)
plt.imshow(img_cinza, cmap = 'gray')
plt.title("Escala cinza")
plt.xticks([]) # Eliminar o eixo X
plt.yticks([])  # #liminar o eixo y
plt.colorbar(orientation = 'horizontal')

plt.subplot(1,2,2)
plt.imshow(img_cinza, cmap = 'jet')
plt.title("JET")
plt.xticks([]) # Eliminar o eixo X
plt.yticks([])  # #liminar o eixo y
plt.colorbar(orientation = 'horizontal')
plt.show()
print('--------------------------------------------------------------------------------------------------------------')
########################################################################################################################

# d) Apresente a imagem em escala de cinza e o seu respectivo histograma; Relacione o histograma e a imagem.
print('1 d)')
# Histograma da imagem
histograma = cv2.calcHist([img_cinza],[0],None,[256],[0,256])
#[img_cinza]: img em escala cinza; [0]: um unico canal (uma matriz); None: mascara (usada p/ selecionar um regiao da
# img); 256 = n. de pontos (0-255); [0,256] = intervalo bits

# Dimensão
dim = len(histograma)
print('Dimensão do Histograma: ' + str(dim))

# Plotar figuras
plt.figure('1d')
plt.subplot(1,2,1)
plt.imshow(img_cinza,cmap="gray")
plt.xticks([])
plt.yticks([])
plt.title("Escala cinza")

plt.subplot(1,2,2)
plt.plot(histograma,color = 'black')
plt.title("Histograma cinza")
plt.xlim([0,256])
plt.xlabel("Valores pixels")
plt.ylabel("Número pixels")

plt.show()
print('--------------------------------------------------------------------------------------------------------------')
########################################################################################################################

# e) Utilizando a imagem em escala de cinza (intensidade) realize a segmentação da imagem de modo a remover o fundo
# da imagem utilizando um limiar manual e o limiar obtido pela técnica de Otsu. Nesta questão apresente o histograma
# com marcação dos limiares utilizados, a imagem limiarizada (binarizada) e a imagem colorida final obtida da
# segmentação. Explique os resultados.
print('1 e)')

# Limiarização - Thresholding
# Limiar manual
limiar_cinza = 130
(L, img_limiar) = cv2.threshold(img_cinza,limiar_cinza,255,cv2.THRESH_BINARY)

(L, img_limiar_inv) = cv2.threshold(img_cinza,limiar_cinza,255,cv2.THRESH_BINARY_INV)
print('Limiar: ' + str(L))

# Plotar figuras - Limiar Manual
plt.figure('Limiar')
plt.subplot(1,4,1)
plt.imshow(img_cinza,cmap='gray')
plt.title('Escala cinza')

plt.subplot(1,4,2)
plt.imshow(img_limiar,cmap='gray')
plt.title('Binário - L: ' + str(limiar_cinza))

plt.subplot(1,4,3)
plt.imshow(img_limiar_inv,cmap='gray')
plt.title('Binário Invertido: L: ' + str(limiar_cinza))

plt.subplot(1,4,4)
plt.plot(histograma,color = 'black')
plt.axvline(x=limiar_cinza,color = 'r')
plt.title("Histograma")
plt.xlim([0,256])
plt.xlabel("Valores pixels")
plt.ylabel("Número pixels")

plt.show()

# Limiar técnica de Otsu
(L,img_otsu) = cv2.threshold(img_cinza,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
(L, img_otsu_inv) = cv2.threshold(img_cinza,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
print('Limiar: ' + str(L))

# Plotar figuras - OTSU
plt.figure('Limiar_Otsu')
plt.subplot(1,4,1)
plt.imshow(img_cinza,cmap='gray')
plt.title('Escala cinza')

plt.subplot(1,4,2)
plt.imshow(img_otsu,cmap='gray')
plt.title('OTSU - L: ' + str(L))

plt.subplot(1,4,3)
plt.imshow(img_otsu_inv,cmap='gray')
plt.title('OTSU Invertido: L: ' + str(L))

plt.subplot(1,4,4)
plt.plot(histograma,color = 'black')
plt.axvline(x=L,color = 'r')
plt.title("Histograma")
plt.xlim([0,256])
plt.xlabel("Valores de pixels")

plt.show()

# Obtendo imagem colorida segmentada
img_segmentada = cv2.bitwise_and(img_cinza,img_cinza,mask=img_limiar_inv)
img_segmentada2 = cv2.bitwise_and(img_cinza,img_cinza,mask=img_otsu_inv)

# Plotar figuras
plt.figure('1e')
plt.subplot(1,3,1)
plt.imshow(img_cinza,cmap='gray')
plt.title('Escala cinza')

plt.subplot(1,3,2)
plt.imshow(img_segmentada)
plt.title('Segmentada manual')

plt.subplot(1,3,3)
plt.imshow(img_segmentada2)
plt.title('Segmentada Otsu')

plt.show()
print('--------------------------------------------------------------------------------------------------------------')
########################################################################################################################

# f) Apresente uma figura contento a imagem selecionada nos sistemas RGB, Lab, HSV e YCrCb.
print('1 f) Imagens em RGB, Lab, HSV e YCrCb')

# Imagem RGB
plt.figure('1f')
plt.subplot(1,4,1)
plt.imshow(img_rgb)
plt.title("RGB")
plt.colorbar(orientation = 'horizontal')

# Imagem Lab
img_Lab = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2Lab)

plt.subplot(1,4,2)
plt.imshow(img_Lab)
plt.title("Lab")
plt.colorbar(orientation = 'horizontal')

# Imagem HSV
img_HSV = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2HSV)
plt.subplot(1,4,3)
plt.imshow(img_HSV)
plt.title("HSV")
plt.colorbar(orientation = 'horizontal')

# Imagem YCrCb
img_YCRCB = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2YCR_CB)
plt.subplot(1,4,4)
plt.imshow(img_YCRCB)
plt.title("YCrCb")
plt.colorbar(orientation = 'horizontal')

plt.show()
print('--------------------------------------------------------------------------------------------------------------')
########################################################################################################################

# g) Apresente uma figura para cada um dos sistemas de cores (RGB, HSV, Lab e YCrCb) contendo a imagem de cada um dos
# canais e seus respectivos histogramas.
print('1 g) Imagens e histogramas em RGB, Lab, HSV e YCrCb')

# Imagem RGB
plt.figure('1g')
plt.subplot(2,4,1)
plt.imshow(img_rgb)
plt.title("RGB")
# Histograma RGB
hist_RGB = cv2.calcHist([img_rgb],[0],None,[256],[0,256])
plt.subplot(2,4,2)
plt.plot(hist_RGB,color = 'black')
plt.title("Histograma RGB")
plt.xlim([0,256])
plt.ylabel("Número de pixels")

# Imagem Lab
img_Lab = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2Lab)
plt.subplot(2,4,5)
plt.imshow(img_Lab)
plt.title("Lab")
# Histograma Lab
hist_Lab = cv2.calcHist([img_Lab],[0],None,[256],[0,256])
plt.subplot(2,4,6)
plt.plot(hist_Lab,color = 'black')
plt.title("Histograma Lab")
plt.xlim([0,256])
plt.xlabel("Valores de pixels")
plt.ylabel("Número de pixels")

# Imagem HSV
img_HSV = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2HSV)
plt.subplot(2,4,3)
plt.imshow(img_HSV)
plt.title("HSV")
# Histograma HSV
hist_HSV = cv2.calcHist([img_HSV],[0],None,[256],[0,256])
plt.subplot(2,4,4)
plt.plot(hist_HSV,color = 'black')
plt.title("Histograma HSV")
plt.xlim([0,256])
plt.ylabel("Número de pixels")

# Imagem YCrCb
img_YCRCB = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2YCR_CB)
plt.subplot(2,4,7)
plt.imshow(img_YCRCB)
plt.title("YCrCb")
# Histograma YCrCb
hist_YCrCb = cv2.calcHist([img_YCRCB],[0],None,[256],[0,256])
plt.subplot(2,4,8)
plt.plot(hist_YCrCb,color = 'black')
plt.title("Histograma YCrCb")
plt.xlim([0,256])
plt.xlabel("Valores de pixels")
plt.ylabel("Número de pixels")

plt.show()
print('--------------------------------------------------------------------------------------------------------------')
########################################################################################################################

# h) Encontre o sistema de cor e o respectivo canal que propicie melhor segmentação da imagem de modo a remover o fundo
# da imagem utilizando limiar manual e limiar obtido pela técnica de Otsu. Nesta questão apresente o histograma com
# marcação dos limiares utilizados, a imagem limiarizada (binarizada) e a imagem colorida final obtida da segmentação.
# Explique os resultados e sua escolha pelo sistema de cor e canal utilizado na segmentação.
print('1 h)')
print('Melhor segmentação: Sistema de cor HSV, canal S')

# Leitura da imagem
img_rc = 'recorte.jpg'
img_bgr = cv2.imread(img_rc,1)
img_HSV = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2HSV)
img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
H,S,V = cv2.split(img_HSV)

# Histograma de imagem
hist_H = cv2.calcHist([img_HSV],[0],None,[256],[0,256])
hist_S = cv2.calcHist([img_HSV],[1],None,[256],[0,256])
hist_V = cv2.calcHist([img_HSV],[2],None,[256],[0,256])

# Limiar manual
limiar_H = 80
(L_H, img_limiar_H) = cv2.threshold(H,limiar_H,255,cv2.THRESH_BINARY)
limiar_S = 90
(L_S, img_limiar_S) = cv2.threshold(S,limiar_S,255,cv2.THRESH_BINARY)
limiar_V = 180
(L_V, img_limiar_V) = cv2.threshold(V,limiar_V,255,cv2.THRESH_BINARY)

# Limiarização - Thresholding - OTSU
(L_H, img_limiar_H) = cv2.threshold(H,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
(L_S, img_limiar_S) = cv2.threshold(S,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
(L_V, img_limiar_V) = cv2.threshold(V,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Plotar imagens - Limiar manual
plt.figure('1.1h')
plt.subplot(3,4,1)
plt.imshow(img_HSV[:,:,0],cmap='gray')
plt.title("H")
plt.xticks([])
plt.yticks([])

plt.subplot(3,4,2)
plt.plot(hist_H,color = 'black')
plt.axvline(x=limiar_H, color = 'blue')
plt.title("Histograma - H")
plt.xlim([0,256])
plt.ylabel("Número pixels")
plt.xticks([])

plt.subplot(3,4,5)
plt.imshow(img_HSV[:,:,1],cmap='gray')
plt.title("S")
plt.xticks([])
plt.yticks([])

plt.subplot(3,4,6)
plt.plot(hist_S,color = 'black')
plt.axvline(x=limiar_S,color = 'red')
plt.title("Histograma - S")
plt.xlim([0,256])
plt.ylabel("Número pixels")
plt.xticks([])

plt.subplot(3,4,9)
plt.imshow(img_HSV[:,:,2],cmap='gray')
plt.title("V")
plt.xticks([])
plt.yticks([])

plt.subplot(3,4,10)
plt.plot(hist_V,color = 'black')
plt.axvline(x=limiar_V,color = 'green')
plt.title("Histograma - V ")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

# Plotar Imagens - Limiar Otsu

plt.subplot(3,4,3)
plt.imshow(img_HSV[:,:,0],cmap='gray')
plt.title("H")
plt.xticks([])
plt.yticks([])

plt.subplot(3,4,4)
plt.plot(hist_H,color = 'black')
plt.axvline(x=L_H, color = 'blue')
plt.title("Histograma - H (Otsu)")
plt.xlim([0,256])
plt.ylabel("Número pixels")
plt.xticks([])

plt.subplot(3,4,7)
plt.imshow(img_HSV[:,:,1],cmap='gray')
plt.title("S")
plt.xticks([])
plt.yticks([])

plt.subplot(3,4,8)
plt.plot(hist_S,color = 'black')
plt.axvline(x=L_S,color = 'red')
plt.title("Histograma - S (Otsu)")
plt.xlim([0,256])
plt.ylabel("Número pixels")
plt.xticks([])

plt.subplot(3,4,11)
plt.imshow(img_HSV[:,:,2],cmap='gray')
plt.title("V")
plt.xticks([])
plt.yticks([])

plt.subplot(3,4,12)
plt.plot(hist_V,color = 'black')
plt.axvline(x=L_V,color = 'green')
plt.title("Histograma - V (Otsu)")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")
plt.show()

# Obter imagem colorida segmentada
img_segm = cv2.bitwise_and(img_HSV,img_HSV,mask=img_limiar_S)
img_segm2 = cv2.bitwise_and(img_rgb,img_rgb,mask=img_limiar_S)

# Plotar imagem colorida segmentada
plt.figure('1.2 h')
plt.subplot(1,4,1)
plt.imshow(img_rgb,cmap="gray")
plt.title("Original")

plt.subplot(1,4,2)
plt.imshow(img_HSV,cmap="gray")
plt.title("HSV")

plt.subplot(1,4,3)
plt.imshow(img_segm)
plt.title('Segmentada HSV')

plt.subplot(1,4,4)
plt.imshow(img_segm2)
plt.title('Segmentada RGB')

plt.show()
print('--------------------------------------------------------------------------------------------------------------')
########################################################################################################################

# i) Obtenha o histograma de cada um dos canais da imagem em RGB, utilizando como mascara a imagem limiarizada
# (binarizada) da letra h.
print('1 i) canais RGB com máscara imagem limiarizada canal S do sistema HSV')

# Histograma de imagem
r,g,b = cv2.split(img_rgb)

(L_S, img_limiar_S) = cv2.threshold(S,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
hist_r = cv2.calcHist([img_rgb],[0],img_limiar_S,[256],[0,256])
hist_g = cv2.calcHist([img_rgb],[1],img_limiar_S,[256],[0,256])
hist_b = cv2.calcHist([img_rgb],[2],img_limiar_S,[256],[0,256])

# Plotar Imagem RGB
plt.figure('1i')
plt.subplot(2,3,1)
plt.imshow(img_rgb[:,:,0],cmap='gray')
plt.title("R")

plt.subplot(2,3,4)
plt.plot(hist_r,color = 'red')
plt.title("Histograma - R")
plt.xlim([0,256])
plt.xlabel("Valores pixels")
plt.ylabel("Número pixels")

plt.subplot(2,3,2)
plt.imshow(img_rgb[:,:,1],cmap='gray')
plt.title("G")

plt.subplot(2,3,5)
plt.plot(hist_g,color = 'green')
plt.title("Histograma - G")
plt.xlim([0,256])
plt.xlabel("Valores pixels")

plt.subplot(2,3,3)
plt.imshow(img_rgb[:,:,2],cmap='gray')
plt.title("B")

plt.subplot(2,3,6)
plt.plot(hist_b,color = 'blue')
plt.title("Histograma - B")
plt.xlim([0,256])
plt.xlabel("Valores pixels")

plt.show()
print('--------------------------------------------------------------------------------------------------------------')
########################################################################################################################

# j) Realize operações aritméticas na imagem em RGB de modo a realçar os aspectos de seu interesse.
# Exemplo (2*R-0.5*G). Explique a sua escolha pelas operações aritméticas.
print('1j: Operação aritmética: (-0.5*B) / R')

# Operação aritmética na imagem RGB
#(-0.5*B)/R
operacao_img = -0.5*img_rgb[:,:,2] / img_rgb[:,:,0]
print(operacao_img)
print('--------------------------------------------')

plt.figure('1j')
plt.subplot(1,3,1)
plt.imshow(img_rgb,cmap='gray')
plt.title("RGB")

plt.subplot(1,3,2)
plt.imshow(operacao_img,cmap='gray')
plt.title("(-0.5*B)/R - Cinza")
plt.colorbar(orientation = 'horizontal')

plt.subplot(1,3,3)
plt.imshow(operacao_img,cmap='jet')
plt.title("(-0.5*B)/R - JET")
plt.colorbar(orientation = 'horizontal')
plt.show()

print('--------------------------------------------------------------------------------------------------------------')
########################################################################################################################


