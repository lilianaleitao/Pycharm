########################################################################################################################
# DATA: 17/07/2020
# DISCIPLINA: VISÃO COMPUTACIONAL NO MELHORAMENTO DE PLANTAS
# ALUNA: LILIANA ROCIVALDA GOMES LEITAO
# GITHUB:
# PROFESSOR: VINÍCIUS QUINTÃO CARNEIRO
# GITHUB: vqcarneiro
########################################################################################################################

# EXERCÍCIO 03:
# a) Crie uma função em um arquivo externo (outro arquivo .py) para calcular a média e a variância amostral de um vetor
# qualquer, baseada em um loop (for).

print('3 a)')
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

