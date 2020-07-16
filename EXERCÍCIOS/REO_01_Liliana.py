########################################################################################################################
# DATA: 17/07/2020
# DISCIPLINA: VISÃO COMPUTACIONAL NO MELHORAMENTO DE PLANTAS
# ALUNA: LILIANA ROCIVALDA GOMES LEITAO
# GITHUB:
# PROFESSOR: VINÍCIUS QUINTÃO CARNEIRO
# GITHUB: vqcarneiro
########################################################################################################################
#'''
# REO 01 - LISTA DE EXERCÍCIOS

# Importação da biblioteca
import numpy as np

print('EXERCÍCIO 01:')
print('1 a)')
# a) Declare os valores 43.5,150.30,17,28,35,79,20,99.07,15 como um array numpy.
# Array numpy: cria arranjos de dados numéricos como os vetores.
# Vetor array numpy
vetor_np = np.array([43.5,150.30,17,28,35,79,20,99.07,15])
print('Vetor array numpy: ' +str(vetor_np))
print('----------------------------------------------------------------------------------------------------')

print('1 b)')
# b) Obtenha as informações de dimensão, média, máximo, mínimo e variância deste vetor.
# Dimensão
dim = len(vetor_np)
print('Dimensão do vetor: ' + str(dim))
# Media
media = np.mean(vetor_np)
print('Média: '+ str(media))
#Máximo
maximo = np.max(vetor_np)
print('Máximo: '+ str(maximo))
#Mínimo
minimo = np.min(vetor_np)
print('Mínimo: '+str(minimo))
#Variância
variancia = np.var(vetor_np)
print('Variância: '+str(variancia))
print('----------------------------------------------------------------------------------------------------')

print('1 c)')
# c) Obtenha um novo vetor em que cada elemento é dado pelo quadrado da diferença entre cada elemento do vetor declarado
# na letra a e o valor da média deste.
# ou seja: (x-media)^2

np.set_printoptions(precision=2) #duas casas decimais depois da virgula
np.set_printoptions(suppress=True)

#Quadrado da diferença
vetor_dois = (vetor_np - np.mean(vetor_np))**2
print('Quadrado da diferença')
print('Novo vetor:' + str(vetor_dois))
print('----------------------------------------------------------------------------------------------------')

print('1 d)')
# d) Obtenha um novo vetor que contenha todos os valores superiores a 30.
vetor_tres = vetor_np>30
print('vetor com valores superiores a 30: ' +str(vetor_tres))
#Novo vetor
print('Novo vetor: ' + str(vetor_np[vetor_tres]))
print('----------------------------------------------------------------------------------------------------')

print('1 e)')
# e) Identifique quais as posições do vetor original possuem valores superiores a 30
import numpy as np
# Vetor original
print('Vetor original: ' +str(vetor_np))
# Valores superiores a 30
vetor_tres = vetor_np>30
print('Valores superiores a 30: ' + str(vetor_np[vetor_tres]))
# Posição valores superiores a 30
pos_sup_30 = np.where(vetor_np>30)
print('Posições com valores superiores a 30: ' + str(pos_sup_30[0]))
print('-----------------------------------------------------------------------------------------------------')

print('1 f)')
# f) Apresente um vetor que contenha os valores da primeira, quinta e última posição.
# Vetor original
print('Vetor original: ' +str(vetor_np))
# Vetor primeira, quinta e última posição
vetor_1_5_8 = vetor_np[[1,5,8]]
print('Vetor da primeira, quinta e última posição: ' + str(vetor_1_5_8))
print('-----------------------------------------------------------------------------------------------------')

print('1 g)')
# g) Crie uma estrutura de repetição usando o for para apresentar cada valor e a sua respectiva posição 
# durante as iterações
# Importação da biblioteca
import time

vetor_np = np.array([43.5,150.30,17,28,35,79,20,99.07,15])
print('Vetor original: ' + str(vetor_np))
print('-----------------------------------------------------------------------------------------------------')
it = 0
for i in range(0,len(vetor_np),1):
    it = it + 1
    print('Na posição ' + str(i) + ' há o elemento: ' + str(vetor_np[int(i)]))
    time.sleep(0.4)

# Obs: função range cria uma sequencia de valores
# (0,len(vetor_np),1): 0 = primeira posição; len(vetor_np) = ultima posição; 1 = passo
# it é o iterador que recebe zero (it = 0) e a cada iteração recebe ele mesmo mais um (it = it+1)
# i é a variavel que recebe a iteração
print('-----------------------------------------------------------------------------------------------------')

print('1 h)')
# h) Crie uma estrutura de repetição usando o for para fazer a soma dos quadrados de cada valor do vetor.

# Vetor original
vetor_np = np.array([43.5,150.30,17,28,35,79,20,99.07,15])
print('Vetor: ' +str(vetor_np))
print('------------------------------------------------------------------------')
it = 0
for sq in vetor_np:
    it = it + 1
    print('Iteração: ' + str(it))
    print('Valor do vetor: ' + str(sq))
    print('Soma de quadrados: '+ str((sq)** 2))
    time.sleep(0.3)
    print('---------------------------------------------------')

print('1 i)')
# i) Crie uma estrutura de repetição usando o while para apresentar todos os valores do vetor

vetor_np = np.array([43.5,150.30,17,28,35,79,20,99.07,15])
print('Vetor: ' + str(vetor_np))
print('-----------------------------------------------------------------------------------------------------')
pos = 0
while vetor_np[pos]!=100:
    print(vetor_np[pos])
    pos = pos+1
    time.sleep(0.4)
    if pos==(len(vetor_np)):
        break

# Obs: pos é o contador, condição do while (similar a iteração do for)
print('----------------------------------------------------------------------------------------------------')

print('1 j)')
# j) Crie um sequência de valores com mesmo tamanho do vetor original e que inicie em 1 e o passo seja também 1.
# Passo: função range (start, stop, step)
# start: inicio da sequencia; stop: final; step: passo
print('Sequencia de valores:' + str(list(range(1,10,1))))
print('Dimensão:' + str(len(range(1,10,1))))

print('----------------------------------------------------------------------------------------------------')

print('1 k)')
# k) Concatene o vetor da letra a com o vetor da letra j.

#Vetor letra a
vetor_np = np.array([43.5,150.30,17,28,35,79,20,99.07,15])
print('vetor letra a :' +str(vetor_np))
print('---------------------------------------------------------------------------------')
#Vetor letra j
vetor_j = (list(range(1,10,1)))
print('vetor letra j :' +str(list(range(1,10,1))))
print('---------------------------------------------------------------------------------')
#Concatenação
concatenação = np.concatenate((vetor_np,vetor_j))
print('concatenação dos vetores (a) e (j):' +str(concatenação))
print('Dimensão: ' + str(len(concatenação)) + ' ' +'elementos')
#'''
########################################################################################################################
########################################################################################################################
########################################################################################################################

'''
# Exercício 02
#a) Declare a matriz abaixo com a biblioteca numpy.
# 1 3 22
# 2 8 18
# 3 4 22
# 4 1 23
# 5 2 52
# 6 2 18
# 7 2 25
print('2 a)')

#Importar biblioteca
import numpy as np
# Array numpy: cria arranjos de dados numéricos como as matrizes.

# Declarando uma matriz
matriz = np.array ([[1,3,22],[2,8,18],[3,4,22],[4,1,23],[5,2,52],[6,2,18],[7,2,25]])
print('Matriz:')
print(matriz)
print('----------------------------------------------------------------------------------------------------')

print('2 b)')
# b) Obtenha o número de linhas e de colunas desta matriz
nl,nc = np.shape(matriz) #shape: obter tamanho de cada dimensão (linhas e colunas)
print('Número de linhas: ' + str(nl))
print('Número de colunas: ' + str(nc))
print('----------------------------------------------------------------------------------------------------')

print('2 c)')
# c) Obtenha as médias das colunas 2 e 3.
# Coluna 2
print('Valores coluna 2: ' + str(matriz[:,1])) #[:,1] = considera todas as linhas (:) e apenas coluna 2 (posiçao 1)
print('Média coluna 2: ' + str(np.mean(matriz[:,1])))
print('------------------------------------------')
# Coluna 3
print('Valores coluna 3: ' + str(matriz[:,2]))
print('Média coluna 3: ' + str(np.mean(matriz[:,2])))
print('----------------------------------------------------------------------------------------------------')

print('2 d)')
# d) Obtenha as médias das linhas considerando somente as colunas 2 e 3

print('Valores das linhas considerando colunas 2 e 3:')
print(str(matriz[:,1:]))
print('média de cada linha')
print('linha 1: ' + str(np.mean(matriz[0,1:])))
print('linha 2: ' + str(np.mean(matriz[1,1:])))
print('linha 3: ' + str(np.mean(matriz[2,1:])))
print('linha 4: ' + str(np.mean(matriz[3,1:])))
print('linha 5: ' + str(np.mean(matriz[4,1:])))
print('linha 6: ' + str(np.mean(matriz[5,1:])))
print('linha 7: ' + str(np.mean(matriz[6,1:])))
print('----------------------------------------------------------------------------------------------------')

print('2 e)')
# e) Considerando que a primeira coluna seja a identificação de genótipos, a segunda nota de severidade de uma doença e
# e a terceira peso de 100 grãos. Obtenha os genótipos que possuem nota de severidade inferior a 5.

# Genótipos
genotipos = matriz[:,0]
print('Genótipos: ' + str(genotipos))
# Notas doença
notas_doença = matriz[:,1]
print('Notas de severidade: ' + str(notas_doença))
# Notas inferiores a 5
notas_inf_5 = matriz[:,1]<5
posiçao = np.where(notas_inf_5)
# Genotipos com notas inferiores a 5
gen_inf_5 = matriz[posiçao]
print('Genótipos com notas inferiores a 5: ' + str(gen_inf_5[:,0]))
print('----------------------------------------------------------------------------------------------------')

print('2 f)')
# f) Considerando que a primeira coluna seja a identificação de genótipos, a segunda nota de severidade de uma doença e
# e a terceira peso de 100 grãos. Obtenha os genótipos que possuem nota de peso de 100 grãos superior ou igual a 22.

# Genótipos
genotipos = matriz[:,0]
print('Genótipos: ' + str(genotipos))
# Peso 100 grãos
peso_100 = matriz[:,2]
print('Peso 100 grãos: ' + str(peso_100))
# Genótipos com notas de peso 100 grãos >= 22
graos_sup_22 = matriz[:,2]>=22
posiçao_2 = np.where(graos_sup_22)
gen_sup_22 = matriz[posiçao_2]
print('Genótipos com notas de peso 100 grãos >= 22: ' + str(gen_sup_22[:,0]))
print('----------------------------------------------------------------------------------------------------')

print('2 g)')
# g) Considerando que a primeira coluna seja a identificação de genótipos, a segunda nota de severidade de uma doença e
# e a terceira peso de 100 grãos. Obtenha os genótipos que possuem nota de severidade igual ou inferior a 3 e peso de 100
# grãos igual ou superior a 22.

# Genótipos
genotipos = matriz[:,0]
print('Genótipos: ' + str(genotipos))
# Notas doença
notas_doença = matriz[:,1]
print('Notas de severidade: ' + str(notas_doença))
# Peso 100 grãos
peso_100 = matriz[:,2]
print('Peso 100 grãos: ' + str(peso_100))
# Genotipos inf <=3 e sup >=22
nota_inf_3 = gen_sup_22[:,1]<=3
posiçao_3 = np.where(nota_inf_3)
gen_inf3_sup22 = gen_sup_22[posiçao_3]
print('Genótipos com notas <=3 e peso 100 grãos >= 22: ' + str(gen_inf3_sup22[:,0]))
print('----------------------------------------------------------------------------------------------------')

print('2 h)')
# h) Crie uma estrutura de repetição com uso do for (loop) para apresentar na tela cada uma das posições da matriz e o seu
#  respectivo valor. Utilize um iterador para mostrar ao usuário quantas vezes está sendo repetido.
#  Apresente a seguinte mensagem a cada iteração "Na linha X e na coluna Y ocorre o valor: Z".
#  Nesta estrutura crie uma lista que armazene os genótipos com peso de 100 grãos igual ou superior a 25

# Obter numero de linhas e colunas da matriz
nl,nc = np.shape(matriz)

# Estrutura de repetição usando o for
# variaveis "m" e "n" correspondem aos loop

import time
contador = 0 #interador

for m in np.arange(0,nl,1): #m = corresponde as linhas; [(0,nl,1)= varia de zero ate nl (7), passo 1]
    for n in np.arange(0,nc,1):
        contador += 1
        print('Iteração: '+ str(contador))
        print('Na linha ' + str(m) + ' e na coluna ' + str(n) + ' ocorre o valor: ' + str(matriz[int(m),int(n)]))
        time.sleep(0.3)
print('----------------------------------------------------------------------------------------------------')
'''
########################################################################################################################
########################################################################################################################
########################################################################################################################

'''
# EXERCÍCIO 03:
print('3 a)')
# a) Crie uma função em um arquivo externo (outro arquivo .py) para calcular a média e a variância amostral de um
# vetor qualquer, baseada em um loop (for).

import numpy as np
import time

vetor = np.array([19,34,78,80,98,99])
print('Vetor: ' +str(vetor))
print('------------------------------------------------------------------------')

it = 0
for el in vetor:
    it = it + 1
    print('Valor do vetor:' + str(el) + '; ' +'Média:' + str(np.mean(vetor)) + '; '+
    'Variância amostral:'+ str(np.var(vetor, ddof=1)))  #ddof = número de graus de liberdade utilizados
    time.sleep(0.3)
print('----------------------------------------------------------------------------------------------------')
'''
'''
print('3 b)')
# b) Simule três arrays com a biblioteca numpy de 10, 100, e 1000 valores e com distribuição normal com média 100 e
# variância 2500. Pesquise na documentação do numpy por funções de simulação.

import numpy as np
np.set_printoptions(precision=2) #duas casas decimais depois da virgula
np.set_printoptions(suppress=True)

# Primeiro array
print('Primeiro array : n = 10; x = 100; DP = 50')
pop1 = np.random.normal(loc=100.0, scale=50.0, size=10)
print(pop1)
print('----------------------------------------------------------------------------------------------------')

# Segundo array
print('Segundo array : n = 100; x = 100; DP = 50')
pop2 = np.random.normal(loc=100.0, scale=50.0, size=100)
print(pop2)
print('----------------------------------------------------------------------------------------------------')

# Terceiro array
print('Terceiro array : n = 1000; x = 100; DP = 50')
pop3 = np.random.normal(loc=100.0, scale=50.0, size=1000)
print(pop3)
print('----------------------------------------------------------------------------------------------------')

print('3 c)')
# c) Utilize a função criada na letra a para obter as médias e variâncias dos vetores simulados na letra b.

import time

vetor_1 = pop1
vetor_2 = pop2
vetor_3 = pop3

it = 0
for el1 in vetor_1:
    it = it + 1
    print('Valor do vetor 1 :' + str(el1) + '; ' + 'Média:' + str(np.mean(vetor_1)) + '; ' +
    'Variância:' + str(np.var(vetor_1)))
    time.sleep(0.1)
print('----------------------------------------------------------------------------------------------------')
it = 0
for el2 in vetor_2:
    it = it + 1
    print('Valor do vetor 2 :' + str(el2) + '; ' + 'Média:' + str(np.mean(vetor_2)) + '; ' +
    'Variância:' + str(np.var(vetor_2)))
    time.sleep(0)
print('----------------------------------------------------------------------------------------------------')
it = 0
for el3 in vetor_3:
    it = it + 1
    print('Valor do vetor 3 :' + str(el3) + '; ' + 'Média:' + str(np.mean(vetor_3)) + '; ' +
    'Variância:' + str(np.var(vetor_3)))
    time.sleep(0)
print('----------------------------------------------------------------------------------------------------')

# d) Crie histogramas com a biblioteca matplotlib dos vetores simulados com valores de 10, 100, 1000 e 100000.
print('3 d)')

import matplotlib.pyplot as plt

pop1 = np.random.normal(loc=100.0, scale=50.0, size=10)
pop2 = np.random.normal(loc=100.0, scale=50.0, size=100)
pop3 = np.random.normal(loc=100.0, scale=50.0, size=1000)
pop4 = np.random.normal(loc=100.0, scale=50.0, size=100000)

# Histograma 1
fig, ax1 = plt.subplots()
ax1.hist(pop1, bins=10, color='darkgreen')
ax1.set_title(r'Histograma Pop 1: $\mu=100$, $\sigma=15$')
# Histograma 2
fig, ax2 = plt.subplots()
ax2.hist(pop2, bins=100, color='red')
ax2.set_title(r'Histograma Pop 2: $\mu=100$, $\sigma=15$')
# Histograma 3
fig, ax3 = plt.subplots()
ax3.hist(pop3, bins=1000, color='blue')
ax3.set_title(r'Histograma Pop 3: $\mu=100$, $\sigma=15$')
# Histograma 4
fig, ax4 = plt.subplots()
ax4.hist(pop3, bins='auto', color='darkviolet')
ax4.set_title(r'Histograma Pop 4: $\mu=100$, $\sigma=15$')
fig.tight_layout()
plt.show()
'''

########################################################################################################################
########################################################################################################################
########################################################################################################################

'''
print('4 a)')
# EXERCÍCIO 04:
# a) O arquivo dados.txt contem a avaliação de genótipos (primeira coluna) em repetições (segunda coluna) quanto a
# quatro variáveis (terceira coluna em diante). Portanto, carregue o arquivo dados.txt com a biblioteca numpy,
# apresente os dados e obtenha as informações de dimensão desta matriz.

import numpy as np
# Considerar duas casas decimais
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

# Carregar arquivo de dados
dados = np.loadtxt('dados.txt')
# Valores dos dados e dimensão da matriz
print('Arquivo dados')
print(dados)
print('----------------------------------------------------------------------------------------------------')
nl,nc = np.shape(dados)
print('Número de Linhas: ' + str(nl))
print('Número de Colunas: ' + str(nc))
print('----------------------------------------------------------------------------------------------------')

print('4 b)')
# b) Pesquise sobre as funções np.unique e np.where da biblioteca numpy

#help (np.unique)
#help (np.where)

print('----------------------------------------------------------------------------------------------------')
print('4 c)')
# c) Obtenha de forma automática os genótipos e quantas repetições foram avaliadas
# Genótipos
genotipos = np.unique(dados[:,[0]]) #[:] considera todas as linhas; [0] primeira coluna na posição zero
print('Genótipos: ' + str(genotipos))
nl,nc = np.shape(dados[:,[0]])
print('Número de linhas: ' + str(nl))
print('Número de colunas: ' + str(nc))
print('-------------------------------------------------------')
# Repetições
repetições = np.unique(dados[:,[1]])
print('Repetições: ' + str(repetições))
nl,nc = np.shape(dados[:,[1]])
print('Número de linhas: ' + str(nl))
print('Número de colunas: ' + str(nc))

print('----------------------------------------------------------------------------------------------------')
print('4 d)')
# d) Apresente uma matriz contendo somente as colunas 1, 2 e 4

dados_sel = dados[:,[0,1,3]] #[:,] = acessar todas as linhas; [0,1,3] = colunas 1 na posição zero, 2 e 4
print('Matriz com as colunas 1, 2 e 4:')
print(dados_sel)
print('----------------------------------------------------------------------------------------------------')

print('4 e)')
# e) Obtenha uma matriz que contenha o máximo, o mínimo, a média e a variância de cada genótipo para a variavel
# da coluna 4. Salve esta matriz em bloco de notas.

# Criar matriz com valores iguais a 0.
matriz_zeros = np.zeros((10,4))  #Tamanho matriz = 10x4, 4 colunas (max,min,med,var)

nl2,nc2 = np.shape(dados)

it=0
genotipo = np.reshape(np.unique(dados[:,0], axis=0),(10,1))
print(genotipo)
for num in np.arange(0,nl2,3):
    matriz_zeros[it, 0] = np.max(dados_sel[num:num + 3, 2], axis=0)
    matriz_zeros[it, 1] = np.min(dados_sel[num:num + 3, 2], axis=0)
    matriz_zeros[it, 2] = np.mean(dados_sel[num:num + 3, 2], axis=0)
    matriz_zeros[it, 3] = np.var(dados_sel[num:num + 3, 2], axis=0)
    it += 1 #incrementa + 1 no it

#print(matriz_zeros)
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

print('Genótipos     Max     Min      Média    Variância')
matriz_concatenada = np.concatenate((genotipo, matriz_zeros),axis=1)
print (matriz_concatenada)

# Salvar os dados
import os
np.savetxt('dados_Q4e.txt', matriz_concatenada, delimiter=' ',newline= os.linesep, fmt='%2.2f')
# Obs:
# dados_Q4e.txt (nome do arquivo a ser salvo);
# matriz_concatenada (matriz a ser salva);
# delimiter=' ' (espaço do delimitador indica dar espaço de uma coluna a outra);
# os.linesep (função para dar espaço de uma linha a outra);
# fmt (configurar o formato do arquivo); %2.2f (2 casas decimais nas demais colunas)
print('----------------------------------------------------------------------------------------------------')

print('4 f)')
# f) Obtenha os genótipos que possuem média (médias das repetições) igual ou superior a 500 da matriz gerada
# na letra anterior.

media_sup_500 = matriz_concatenada[matriz_concatenada[:,3]>=500]
print('Genotipos com médias iguais ou superiore a 500: ' + str(media_sup_500[:,0])) #[:] todas as linhas
                                                                                    #[0] coluna 1, posiçao zero
print('----------------------------------------------------------------------------------------------------')

print('4 g)')
# g) Apresente os seguintes graficos:
print('Médias dos genótipos para cada variável')
#Utilizar o comando plt.subplot para mostrar mais de um grafico por figura

nl3,nc3 = np.shape(dados)
matriz_de_zeros = np.zeros((10,5))

Genotipo = np.reshape(np.unique(dados[:,0], axis=0),(10,1))
it=0
for g in np.arange(0, nl3, 3): # nl3: percorre as 30 linhas da matriz para as 3 repetições
    matriz_de_zeros[it, 0] = np.mean(dados[g:g + 3, 2], axis=0)
    matriz_de_zeros[it, 1] = np.mean(dados[g:g + 3, 3], axis=0)
    matriz_de_zeros[it, 2] = np.mean(dados[g:g + 3, 4], axis=0)
    matriz_de_zeros[it, 3] = np.mean(dados[g:g + 3, 5], axis=0)
    matriz_de_zeros[it, 4] = np.mean(dados[g:g + 3, 6], axis=0)
    it = it + 1

matriz_concatenada2 = np.concatenate((Genotipo, matriz_de_zeros),axis=1)
print(matriz_concatenada2)

from matplotlib import pyplot as plt

plt.figure('Média dos genótipos para cada variável')
plt.subplot(2,3,1)
plt.bar(x=matriz_concatenada2[:,0],height=matriz_concatenada2[:,1], width=0.6, color='darkgreen')
plt.title('Variável 1')
plt.ylabel('Média')

plt.subplot(2,3,2)
plt.bar(x=matriz_concatenada2[:,0],height=matriz_concatenada2[:,2], width=0.6, color='purple')
plt.title('Variável 2')
plt.ylabel('Média')

plt.subplot(2,3,3)
plt.bar(x=matriz_concatenada2[:,0],height=matriz_concatenada2[:,3], width=0.6, color='red')
plt.title('Variável 3')
plt.ylabel('Média')

plt.subplot(2,3,4)
plt.bar(x=matriz_concatenada2[:,0],height=matriz_concatenada2[:,4], width=0.6, color='blue')
plt.title('Variável 4')
plt.ylabel('Média')

plt.subplot(2,3,5)
plt.bar(x=matriz_concatenada2[:,0],height=matriz_concatenada2[:,5], width=0.6, color='orange')
plt.title('Variável 5')
plt.ylabel('Média')
plt.legend()
plt.show()

print('Dispersão 2D das médias dos genótipos')
# Utilizar as três primeiras variáveis). No eixo X uma variável e no eixo Y outra.

cores_usadas = ['black','blue','red','green','yellow','pink','gray','orange','purple','black']
plt.style.use('ggplot')
fig = plt.figure('Dispersão 2D')

nl4,nc4 = np.shape(matriz_concatenada2)

plt.subplot(1,3,1)
for i in np.arange(0,nl4,1):
    plt.scatter(matriz_concatenada2[i,1], matriz_concatenada2[i,2], s=50, alpha=0.8, label = matriz_concatenada2[i,0], c = cores_usadas[i])
plt.title('Dispersão 1')
plt.xlabel('Variavel 1')
plt.ylabel('Variavel 2')

plt.subplot(1,3,2)
for i in np.arange(0,nl4,1):
    plt.scatter(matriz_concatenada2[i,1], matriz_concatenada2[i,3], s=50, alpha=0.8, label = matriz_concatenada2[i,0], c = cores_usadas[i])
plt.title('Dispersão 2')
plt.xlabel('Variavel 1')
plt.ylabel('Variavel 3')

plt.subplot(1,3,3)
for i in np.arange(0,nl4,1):
    plt.scatter(matriz_concatenada2[i,2], matriz_concatenada2[i,3], s=50, alpha=0.8, label = matriz_concatenada2[i,0], c = cores_usadas[i])
plt.title('Dispersão 3')
plt.xlabel('Variavel 2')
plt.ylabel('Variavel 3')
plt.legend()
plt.show()
'''
########################################################################################################################
########################################################################################################################
########################################################################################################################