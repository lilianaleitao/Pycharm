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
from skimage.measure import label, regionprops
########################################################################################################################
# Leitura da imagem
Arquivo = "batata_doce.jpeg"
# Nome do arquivo a ser utilizado na análise
img_bgr = cv2.imread(Arquivo,1) # Carrega imagem (1 - Colorida (BGR))
img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
########################################################################################################################
# a) Aplique o filtro de média com cinco diferentes tamanhos de kernel e compare os resultados com a imagem original.
print('1 a) kernel usados: 11x11; 25x25; 33x33; 47x47 e 59x59')

# Filtro de média
img_filtro_media1 = cv2.blur(img_rgb,(11,11)) # kernel 11x11 (11 linhas e 11 colunas)
img_filtro_media2 = cv2.blur(img_rgb,(25,25))
img_filtro_media3 = cv2.blur(img_rgb,(33,33))
img_filtro_media4 = cv2.blur(img_rgb,(47,47))
img_filtro_media5 = cv2.blur(img_rgb,(59,59))

# Plotar imagens
plt.figure('1a')
plt.subplot(2,3,1)
plt.imshow(img_rgb)
plt.xticks([])
plt.yticks([])
plt.title("Original")

plt.subplot(2,3,2)
plt.imshow(img_filtro_media1)
plt.xticks([])
plt.yticks([])
plt.title("Kernel 11x11")

plt.subplot(2,3,3)
plt.imshow(img_filtro_media2)
plt.xticks([])
plt.yticks([])
plt.title("Kernel 25x25")

plt.subplot(2,3,4)
plt.imshow(img_filtro_media3)
plt.xticks([])
plt.yticks([])
plt.title("Kernel 33x33")

plt.subplot(2,3,5)
plt.imshow(img_filtro_media4)
plt.xticks([])
plt.yticks([])
plt.title("Kernel 47x47")

plt.subplot(2,3,6)
plt.imshow(img_filtro_media5)
plt.xticks([])
plt.yticks([])
plt.title("Kernel 59x59")

plt.show()

print('--------------------------------------------------------------------------------------------------------------')
######################################################################################################################

# b) Aplique diferentes tipos de filtros com pelo menos dois tamanhos de kernel e compare os resultados entre si e
# com a imagem original.
print('1 b) Filtros: media, gaussiano, mediana e bilateral; kernel: 27x27 e 47x47')

# Filtro de média (media aritmetica)
filtro_media1 = cv2.blur(img_rgb,(27,27))
filtro_media2 = cv2.blur(img_rgb,(47,47))
# obs: parametros 1 e 2 = kernel 27x27 (27 linhas e 27 colunas)

# Filtro Gaussiano (média ponderada)
# média ponderada dos pixels vizinhos ao pixel de interesse, de modo que pixels mais proximos recebem pesos
# maiores e pixels mais distantes pesos menores.
filtro_gaussiano1 = cv2.GaussianBlur(img_rgb,(27,27),0)
filtro_gaussiano2 = cv2.GaussianBlur(img_rgb,(47,47),0)
# obs: parametros 1 e 2 = kernel
# obs: parametro 3 = zero, os pesos são determinados automaticamente pela função Blur

# Filtro de mediana
# mediana dos pixels na região de vizinhança do pixel de interesse
filtro_mediana1 = cv2.medianBlur(img_rgb,27)
filtro_mediana2 = cv2.medianBlur(img_rgb,47)

# Filtro bilateral
# Identificação das bordas dos objetos, sendo eficaz na remoção de ruído.
filtro_bilateral1 = cv2.bilateralFilter(img_rgb,27,27,3)
filtro_bilateral2 = cv2.bilateralFilter(img_rgb,47,47,3)
# obs: parametro 1 = kernel
# obs: parametro 2 = desvio padrão, elimina os pixels discrepantes da imagem (outliers)
# obs: parametro 3 = ponderação de pesos das distancias dos pixels vizinhos

# Plotar imagens
plt.figure('1b')
plt.subplot(3,4,1)
plt.imshow(img_rgb)
plt.xticks([])
plt.yticks([])
plt.title("Original")

plt.subplot(3,4,5)
plt.imshow(filtro_media1)
plt.xticks([])
plt.yticks([])
plt.title("Filtro média (27x27)")

plt.subplot(3,4,6)
plt.imshow(filtro_gaussiano1)
plt.xticks([])
plt.yticks([])
plt.title("Filtro Gaussiano (27x27)")

plt.subplot(3,4,7)
plt.imshow(filtro_mediana1)
plt.xticks([])
plt.yticks([])
plt.title("Filtro mediana (27x27)")

plt.subplot(3,4,8)
plt.imshow(filtro_bilateral1)
plt.xticks([])
plt.yticks([])
plt.title("Filtro bilateral (27x27)")

plt.subplot(3,4,9)
plt.imshow(filtro_media2)
plt.xticks([])
plt.yticks([])
plt.title("Filtro média (47x47)")

plt.subplot(3,4,10)
plt.imshow(filtro_gaussiano2)
plt.xticks([])
plt.yticks([])
plt.title("Filtro Gaussiano (47x47)")

plt.subplot(3,4,11)
plt.imshow(filtro_mediana2)
plt.xticks([])
plt.yticks([])
plt.title("Filtro mediana (47x47)")

plt.subplot(3,4,12)
plt.imshow(filtro_bilateral2)
plt.xticks([])
plt.yticks([])
plt.title("Filtro bilateral (47x47)")

plt.show()

print('--------------------------------------------------------------------------------------------------------------')
######################################################################################################################

# c) Realize a segmentação da imagem utilizando o processo de limiarização. Utilizando o reconhecimento de contornos,
# identifique e salve os objetos de interesse. Além disso, acesse as bibliotecas Opencv e Scikit-Image, verifique as
# variáveis que podem ser mensuradas e extraia as informações pertinentes (crie e salve uma tabela com estes dados).
# Apresente todas as imagens obtidas ao longo deste processo.
print('1 c)')

# Leitura da imagem RGB
r,g,b = cv2.split(img_rgb)

# Limiarização - OTSU
(L_r, img_limiar_r) = cv2.threshold(r,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
(L_g, img_limiar_g) = cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
(L_b, img_limiar_b) = cv2.threshold(b,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
(L_b, img_limiar_b_inv) = cv2.threshold(b,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Segmentação da imagem
img_seg_b = cv2.bitwise_and(img_rgb,img_rgb,mask=img_limiar_b)
img_seg_b_inv = cv2.bitwise_and(img_rgb,img_rgb,mask=img_limiar_b_inv)

# Plotar imagens
print('Melhor segmentação: Sistema de cor RGB, canal B, Otsu invertido')
plt.figure('1c segmentada')
plt.subplot(2,3,1)
plt.imshow(img_limiar_r,cmap='gray')
plt.title('R - OTSU')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,2)
plt.imshow(img_limiar_g,cmap='gray')
plt.title('G - OTSU')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,3)
plt.imshow(img_limiar_b,cmap='gray')
plt.title('B - OTSU')
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,4)
plt.imshow(img_rgb)
plt.xticks([])
plt.yticks([])
plt.title("Original")

plt.subplot(2,3,5)
plt.imshow(img_seg_b)
plt.xticks([])
plt.yticks([])
plt.title('Segmentada')

plt.subplot(2,3,6)
plt.imshow(img_seg_b_inv)
plt.xticks([])
plt.yticks([])
plt.title('Segmentada invertida')
plt.show()
#-------------------------------------------------------------------------

# Leitura da imagem YCrCb
img_YCrCb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2YCR_CB)
# Partição dos canais
Y,Cr,Cb = cv2.split(img_YCrCb)
Cb = cv2.medianBlur(Cb,9) # canal Cb recebe filtro mediana com kernel 9
# Limiarização - OTSU
(L, img_limiar_inv) = cv2.threshold(Cb,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# Segmentação da imagem
img_segmentada = cv2.bitwise_and(img_rgb,img_rgb,mask=img_limiar_inv)
# Objeto
mascara = img_limiar_inv.copy()
# Contorno
cnts,h = cv2.findContours(mascara, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print('-----------------------------------------------------------------')

for (i, c) in enumerate(cnts):
	(x, y, w, h) = cv2.boundingRect(c)
	obj = img_limiar_inv[y:y+h,x:x+w]
	obj_rgb = img_segmentada[y:y+h,x:x+w]
	obj_bgr = cv2.cvtColor(obj_rgb,cv2.COLOR_RGB2BGR)
	cv2.imwrite('bd'+str(i+1)+'.png',obj_bgr)
	cv2.imwrite('bdb'+str(i+1)+'.png',obj)

# https://scikit-image.org/docs/dev/api/skimage.measure.html
	regiao = regionprops(obj)
	print('Corte da batata doce: ', str(i+1))
	print('Dimensão da imagem: ', np.shape(obj))
	print('Mensurações')
	print('Centroide: ', regiao[0].centroid)
	print('Comprimento do eixo menor: ', regiao[0].minor_axis_length)
	print('Comprimento do eixo maior: ', regiao[0].major_axis_length)
	print('Razão: ', regiao[0].major_axis_length / regiao[0].minor_axis_length)
	area = cv2.contourArea(c)
	print('Área: ', area)
	print('Perímetro: ', cv2.arcLength(c,True))

	print('Medidas de Cor')
	min_val_r, max_val_r, min_loc_r, max_loc_r = cv2.minMaxLoc(obj_rgb[:,:,0], mask=obj)
	print('Valor Mínimo no R: ', min_val_r, ' - Posição: ', min_loc_r)
	print('Valor Máximo no R: ', max_val_r, ' - Posição: ', max_loc_r)
	med_val_r = cv2.mean(obj_rgb[:,:,0], mask=obj)
	print('Média no Vermelho: ', med_val_r)

	min_val_g, max_val_g, min_loc_g, max_loc_g = cv2.minMaxLoc(obj_rgb[:, :, 1], mask=obj)
	print('Valor Mínimo no G: ', min_val_g, ' - Posição: ', min_loc_g)
	print('Valor Máximo no G: ', max_val_g, ' - Posição: ', max_loc_g)
	med_val_g = cv2.mean(obj_rgb[:,:,1], mask=obj)
	print('Média no Verde: ', med_val_g)

	min_val_b, max_val_b, min_loc_b, max_loc_b = cv2.minMaxLoc(obj_rgb[:, :, 2], mask=obj)
	print('Valor Mínimo no B: ', min_val_b, ' - Posição: ', min_loc_b)
	print('Valor Máximo no B: ', max_val_b, ' - Posição: ', max_loc_b)
	med_val_b = cv2.mean(obj_rgb[:,:,2], mask=obj)
	print('Média no Azul: ', med_val_b)

seg = img_segmentada.copy()
cv2.drawContours(seg, cnts, -1, (0, 255, 0), 2)

print('Número de discos de batata doce : ' + str(len(cnts)))
print('-----------------------------------------------------------------')

# Plotar imagens
plt.figure('1c mensuração')
plt.subplot(1,2,1)
plt.imshow(seg)
plt.xticks([])
plt.yticks([])
plt.title('Original')

plt.subplot(1,2,2)
plt.imshow(obj_bgr[:,:,0])
plt.xticks([])
plt.yticks([])
plt.title('Disco ')
plt.show()

# Salvar 
import pandas as pd
mask = np.zeros(img_rgb.shape,dtype = np.uint8)
#mascara=img_seg_b_inv.copy()
cnts = cv2.findContours(mascara,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

dimen = []
for (i,c) in enumerate(cnts):
	(x,y,w,h) = cv2.boundingRect(c)
	obj = img_segmentada[y:y+h,x:x+w]
	obj_bgr = cv2.cvtColor(obj,cv2.COLOR_RGB2BGR)
	cv2.imwrite(f'Batata{i+1}.png',obj_bgr)
	area = cv2.contourArea(c)
	razao = round(h/w,2)
	min_r = min_val_r
	max_r = max_val_r
	min_g = min_val_g
	max_g = max_val_g
	min_b = min_val_b
	max_b = max_val_b
	dimen += [[str(i+1),str(h),str(w),str(area),str(razao)]]

dados_discos = pd.DataFrame(dimen)
dados_discos = dados_discos.rename(columns={0:'Disco', 1: 'Altura',2:'Largura',3:'Razao',4:'Area'})
dados_discos.to_csv('Questao1c.csv', index=False, sep=';')
######################################################################################################################

# d) Utilizando máscaras, apresente o histograma somente dos objetos de interesse.
print('1 d)')

plt.figure('1d')
for (i, c) in enumerate(cnts):

	(x,y,w,h) = cv2.boundingRect(c)
	print('Corte batata doce #%d' % (i+1))
	print(cv2.contourArea(c))
	obj = img_limiar_inv[y:y+h,x:x+w]
	obj_rgb = img_segmentada[y:y+h,x:x+w]

	grafico = True
	if grafico == True:

		hist_r = cv2.calcHist([obj_rgb[:,:,0]],[0],obj,[256],[0,256])
		hist_g = cv2.calcHist([obj_rgb[:,:,1]],[0],obj,[256],[0, 256])
		hist_b = cv2.calcHist([obj_rgb[:,:,2]],[0],obj,[256],[0, 256])
		#obj = img_rgb[y:y + h, x:x + w]

		plt.subplot(2,3,1)
		plt.imshow(obj_rgb[:,:,0],cmap='gray')
		plt.xticks([])
		plt.yticks([])
		plt.title('Disco: ' + str(i + 1))

		plt.subplot(2,3,2)
		plt.imshow(obj_rgb[:,:,1],cmap='gray')
		plt.xticks([])
		plt.yticks([])
		plt.title('Disco: ' + str(i + 1))

		plt.subplot(2,3,3)
		plt.imshow(obj_rgb[:,:,2],cmap='gray')
		plt.xticks([])
		plt.yticks([])
		plt.title('Disco: ' + str(i + 1))

		plt.subplot(2,3,4)
		plt.plot(hist_r, color='r')
		plt.title("Histograma - R")
		plt.xlim([0, 256])
		plt.xlabel("Valores Pixels")
		plt.ylabel("Número de Pixels")

		plt.subplot(2,3,5)
		plt.plot(hist_g, color='green')
		plt.title("Histograma - G")
		plt.xlim([0, 256])
		plt.xlabel("Valores Pixels")
		plt.ylabel("Número de Pixels")

		plt.subplot(2,3,6)
		plt.plot(hist_r, color='blue')
		plt.title("Histograma - B")
		plt.xlim([0, 256])
		plt.xlabel("Valores Pixels")
		plt.ylabel("Número de Pixels")
		plt.show()
######################################################################################################################

# e) Realize a segmentação da imagem utilizando a técnica de k-means. Apresente as imagens obtidas neste processo.
print('1 e) Segmentação k-means')

print('Dimensão: ',np.shape(img_rgb))
print(np.shape(img_rgb)[0], ' x ',np.shape(img_rgb)[1], ' = ', np.shape(img_rgb)[0] * np.shape(img_rgb)[1])
print('--------------------------------------------------------------------------------------------------------------')

# Formatação da imagem para uma matriz de dados
pixel_values = img_rgb.reshape((-1, 3))
# Conversão para Decimal
pixel_values = np.float32(pixel_values)
print('Dimensão Matriz: ',pixel_values.shape)
print('--------------------------------------------------------------------------------------------------------------')

# K-means
# Critério de Parada
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
# Número de Grupos (k)
k = 2
dist, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 30, cv2.KMEANS_RANDOM_CENTERS)
print('SQ das Distâncias de Cada Ponto ao Centro: ', dist)
print('Dimensão labels: ', labels.shape)
print('Valores únicos: ',np.unique(labels))
print('Tipo labels: ', type(labels))
print('--------------------------------------------------------------------------------------------------------------')

# flatten the labels array
labels = labels.flatten()
print('Dimensão flatten labels: ', labels.shape)
print('Tipo labels (f): ', type(labels))
print('--------------------------------------------------------------------------------------------------------------')

# Valores dos labels
val_unicos,contagens = np.unique(labels,return_counts=True)
val_unicos = np.reshape(val_unicos,(len(val_unicos),1))
contagens = np.reshape(contagens,(len(contagens),1))
# Histograma
hist = np.concatenate((val_unicos,contagens),axis=1)
print('Histograma')
print(hist)
print('--------------------------------------------------------------------------------------------------------------')
print('Centroides Decimais')
print(centers)
print('--------------------------------------------------------------------------------------------------------------')

# Conversão dos centroides para valores de interos de 8 digitos
centers = np.uint8(centers)
print('Centroides uint8')
print(centers)
print('--------------------------------------------------------------------------------------------------------------')

# Conversão dos pixels para a cor dos centroides
matriz_segmentada = centers[labels]
print('Dimensão Matriz Segmentada: ',matriz_segmentada.shape)
print('Matriz Segmentada')
print(matriz_segmentada[0:5,:])
print('--------------------------------------------------------------------------------------------------------------')

# Reformatar a matriz na imagem de formato original
img_segmentada = matriz_segmentada.reshape(img_rgb.shape)

# Grupo 1
original_01 = np.copy(img_rgb)
matriz_or_01 = original_01.reshape((-1, 3))
matriz_or_01[labels != 0] = [0, 0, 0]
img_final_01 = matriz_or_01.reshape(img_rgb.shape)

# Grupo 2
original_02 = np.copy(img_rgb)
matriz_or_02 = original_02.reshape((-1, 3))
matriz_or_02[labels != 1] = [0, 0, 0]
img_final_02 = matriz_or_02.reshape(img_rgb.shape)

# Plotar as imagens
plt.figure('1e')
plt.subplot(2,2,1)
plt.imshow(img_rgb)
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,2)
plt.imshow(img_segmentada)
plt.title('Rotulos')
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,3)
plt.imshow(img_final_01)
plt.title('Grupo 1')
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,4)
plt.imshow(img_final_02)
plt.title('Grupo 2')
plt.xticks([])
plt.yticks([])

plt.show()
########################################################################################################################

# f) Realize a segmentação da imagem utilizando a técnica de watershed. Apresente as imagens obtidas neste processo.
print('1 f) Segmentação watershed')

# Importar pacotes
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage

# Leitura da imagem YCrCb
img_YCrCb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2YCR_CB)
# Partição dos canais
Y,Cr,Cb = cv2.split(img_YCrCb)
# Limiarização - OTSU
limiar, mascara = cv2.threshold(Cr,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Distancia entre pontos brancos da mascara e pontos pretos do fundo (zero)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html
img_dist = ndimage.distance_transform_edt(mascara)
# Calcula a distância euclidiana do pixel de fundo para cada pixel de primeiro plano.

# Localização e separação dos picos
# https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.peak_local_max
max_local = peak_local_max(img_dist, indices=False, min_distance=200,
	labels=mascara)
# Retorna uma matriz booleana com os picos da imagem baseados nas distâncias
# min_distance: Número mínimo de pixels que separam os picos
print('----------------------------------------------------------------')

print('Número de Picos')
print(np.unique(max_local,return_counts=True))
print('----------------------------------------------------------------')

# Marcação dos picos
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html
# https://en.wikipedia.org/wiki/Connected-component_labeling
marcadores,n_marcadores = ndimage.label(max_local, structure=np.ones((3, 3)))
print('Análise de conectividade - Marcadores')
print(np.unique(marcadores,return_counts=True))
print('----------------------------------------------------------------')

# Imagem rotulada
# https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.watershed
# https://en.wikipedia.org/wiki/Watershed_(image_processing)
# https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html
img_ws = watershed(-img_dist, marcadores, mask=mascara)
print('Imagem Segmentada - Watershed')
print(np.unique(img_ws,return_counts=True))
print("Número de Batatas: ", len(np.unique(img_ws)) - 1)
print('--------------------------------------------------------------------------------------------------------------')

# Seleção de imagem
img_final_1 = np.copy(img_rgb)
img_final_1[img_ws != 1] = [0,0,0] # Acessando a batata doce 1
img_final_2 = np.copy(img_rgb)
img_final_2[img_ws != 2] = [0,0,0] # Acessando a batata doce 2
img_final_3 = np.copy(img_rgb)
img_final_3[img_ws != 3] = [0,0,0] # Acessando a batata doce 3

# Plotar imagens
plt.figure('1f')
plt.subplot(2,4,1)
plt.imshow(img_rgb)
plt.xticks([])
plt.yticks([])
plt.title('Original')

plt.subplot(2,4,2)
plt.imshow(Cr,cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('Cr')

plt.subplot(2,4,3)
plt.imshow(mascara,cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('Mascara')

plt.subplot(2,4,4)
plt.imshow(img_dist,cmap='jet')
plt.xticks([])
plt.yticks([])
plt.title('Distância')

plt.subplot(2,4,5)
plt.imshow(img_ws,cmap='jet')
plt.xticks([])
plt.yticks([])
plt.title('Batata doce')

plt.subplot(2,4,6)
plt.imshow(img_final_1)
plt.xticks([])
plt.yticks([])
plt.title('Disco 1')

plt.subplot(2,4,7)
plt.imshow(img_final_2)
plt.xticks([])
plt.yticks([])
plt.title('Disco 2')

plt.subplot(2,4,8)
plt.imshow(img_final_3)
plt.xticks([])
plt.yticks([])
plt.title('Disco 3')

plt.show()
########################################################################################################################

# g) Compare os resultados das três formas de segmentação (limiarização, k-means e watershed) e identifique as
# potencialidades de cada delas.
print('1 g) Segmentação: limiarização, k-means e watershed')

# Limiarização
plt.figure('1g')
plt.subplot(2,2,1)
plt.imshow(img_rgb)
plt.xticks([])
plt.yticks([])
plt.title("Original")

plt.subplot(2,2,2)
plt.imshow(img_seg_b_inv)
plt.xticks([])
plt.yticks([])
plt.title('Limiarização')

plt.subplot(2,2,3)
plt.imshow(img_final_02)
plt.xticks([])
plt.yticks([])
plt.title('K-means')

plt.subplot(2,2,4)
plt.imshow(img_final_1)
plt.xticks([])
plt.yticks([])
plt.title('Watershed')

plt.show()
########################################################################################################################
