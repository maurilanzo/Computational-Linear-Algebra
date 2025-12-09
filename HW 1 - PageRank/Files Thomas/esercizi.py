# %% [BLOCCO 1] LIBRERIA FUNZIONI
# Qui definiamo gli strumenti matematici descritti nel paper.
import numpy as np

import get_google_matrix , power_method , print_ranking from metodi

# %% [BLOCCO 2] GRAFO 1 (Figura 2.1 del paper)
# Ricostruzione manuale della matrice A.
# Regola: Se la pagina J linka a I, allora A[i,j] = 1 / (num link uscenti da J)

# COLONNA 1 (Pag 1): Linka a 2, 3, 4 -> 1/3 a ciascuno
# COLONNA 2 (Pag 2): Linka a 3, 4    -> 1/2 a ciascuno
# COLONNA 3 (Pag 3): Linka a 1       -> 1/1 a pag 1
# COLONNA 4 (Pag 4): Linka a 1, 3    -> 1/2 a ciascuno

#     P1   P2   P3   P4
A1 = np.array([
    [0.0, 0.0, 1.0, 0.5], # Verso Pagina 1
    [1/3, 0.0, 0.0, 0.0], # Verso Pagina 2
    [1/3, 0.5, 0.0, 0.5], # Verso Pagina 3
    [1/3, 0.5, 0.0, 0.0]  # Verso Pagina 4
])

print("Matrice A (Grafo 1):")
print(A1)

# Trasformiamo in Matrice Google M e calcoliamo
M1 = get_google_matrix(A1, m=0.15)
scores1, iters1 = power_method(M1)
print_ranking(scores1, "Risultati Grafo 1")


# %% [BLOCCO 3] GRAFO 2 (Figura 2.2 - Disconnesso)
# Due isole separate: {1,2} e {3,4,5}

#     P1   P2   P3   P4   P5
A2 = np.array([
    [0.0, 1.0, 0.0, 0.0, 0.0], # 1 riceve da 2
    [1.0, 0.0, 0.0, 0.0, 0.0], # 2 riceve da 1
    [0.0, 0.0, 0.0, 0.5, 0.5], # 3 riceve da 4 e 5
    [0.0, 0.0, 1.0, 0.0, 0.5], # 4 riceve da 3 e 5
    [0.0, 0.0, 0.0, 0.0, 0.0]  # 5 non riceve da nessuno (ma linka a 3 e 4)
])
# Nota: La colonna 5 si divide tra riga 3 e riga 4. 
# La riga 5 è tutta 0 perché nessuno linka a 5.

M2 = get_google_matrix(A2, m=0.15)
scores2, iters2 = power_method(M2)
print_ranking(scores2, "Risultati Grafo 2 (Disconnesso)")


# %% [BLOCCO 4] ESERCIZIO 4 (Dangling Node)
# Partiamo dal Grafo 1, ma RIMUOVIAMO il link da 3 a 1.
# La pagina 3 ora non linka a nessuno.

#     P1   P2   P3   P4
A_ex4 = np.array([
    [0.0, 0.0, 0.0, 0.5], # P3 -> P1 rimosso (era 1.0, ora 0.0)
    [1/3, 0.0, 0.0, 0.0], 
    [1/3, 0.5, 0.0, 0.5], 
    [1/3, 0.5, 0.0, 0.0] 
])
# Nota: La colonna 3 ora è TUTTA ZERI. È substocastica.

print("\n--- Analisi Esercizio 4 (Dangling Node) ---")
# Usiamo gli autovalori diretti (numpy.linalg.eig) perché il power method classico
# su matrici substocastiche tende a svanire a 0.
eigvals, eigvecs = np.linalg.eig(A_ex4)

# Prendiamo solo la parte reale e cerchiamo il massimo
real_eigvals = eigvals.real
idx_max = np.argmax(real_eigvals)
lambda_perron = real_eigvals[idx_max]
v_perron = eigvecs[:, idx_max].real

# Normalizziamo
if v_perron.sum() < 0: v_perron = -v_perron # Correggi segno se negativo
v_perron = v_perron / v_perron.sum()

print(f"Autovalore di Perron (lambda): {lambda_perron:.4f}")
print_ranking(v_perron, "Ranking Esercizio 4")


# %% [BLOCCO 5] ESERCIZIO 11 (Aggiunta Pagina 5)
# Modifichiamo il Grafo 1 aggiungendo P5.
# - P5 linka a P3.
# - P3 linka a P5 (oltre che a P1, come faceva prima).

#     P1   P2   P3   P4   P5
A_ex11 = np.array([
    # Col 3 modificata: prima dava 1.0 a P1, ora divide tra P1 e P5
    [0.0, 0.0, 0.5, 0.5, 0.0], # P1 riceve 0.5 da P3, 0.5 da P4
    [1/3, 0.0, 0.0, 0.0, 0.0], # P2 (invariato)
    [1/3, 0.5, 0.0, 0.5, 1.0], # P3 riceve da 1, 2, 4 E ORA DA 5 (1.0)
    [1/3, 0.5, 0.0, 0.0, 0.0], # P4 (invariato)
    [0.0, 0.0, 0.5, 0.0, 0.0]  # P5 riceve 0.5 da P3
])

print("\n--- Analisi Esercizio 11 (Pagina 5 aggiunta) ---")
print("Nuova Matrice A (5x5):")
print(A_ex11)

M_ex11 = get_google_matrix(A_ex11, m=0.15)
scores_ex11, iters_ex11 = power_method(M_ex11)
print_ranking(scores_ex11, "Risultati Esercizio 11")