import numpy as np
import time

def get_google_matrix(A, m=0.15):
    """
    Applica la formula (3.1) del paper: M = (1-m)A + mS
    A: Matrice dei link (stocastica o substocastica)
    m: Damping factor (probabilità che l'utente si annoi e cambi sito a caso)
    """
    n = A.shape[0]
    # S è la matrice dove ogni cella vale 1/n (democrazia totale)
    S = np.ones((n, n)) / n
    M = (1 - m) * A + m * S
    return M

def power_method(M, tol=1e-9, max_iter=1000):
    """
    Trova l'autovettore dominante (PageRank) moltiplicando ripetutamente.
    Equivale alla simulazione di un utente che naviga all'infinito.
    """
    n = M.shape[0]
    x = np.ones(n) / n  # Iniziamo con probabilità uguale per tutti
    
    for k in range(max_iter):
        x_new = np.dot(M, x) # Moltiplicazione matrice-vettore
        
        # Controlliamo se è cambiato poco rispetto al giro prima
        if np.linalg.norm(x_new - x, 1) < tol:
            return x_new, k + 1
        x = x_new
            
    return x, max_iter

def print_ranking(scores, title="Ranking"):
    print(f"\n--- {title} ---")
    for i, s in enumerate(scores):
        print(f"Pagina {i+1}: {s:.5f}")
        
        
class PageRankEngine:
    def __init__(self, damping_factor=0.85, tolerance=1e-9, max_iter=100):
       
        self.d = damping_factor      # 85% di probabilità di seguire i link
        self.m = 1.0 - self.d        # 15% di probabilità di teletrasporto
        self.tol = tolerance         # Quando smettere (precisione)
        self.max_iter = max_iter     # Limite di sicurezza giri
        
        self.urls = {}           # Mappa ID -> Nome Sito
        self.links = {}          # Mappa ID -> Lista di chi linko (es. 1 -> [2, 3])
        self.out_degree = {}     # Mappa ID -> Quanti link ho in uscita
        self.num_nodes = 0       # Contatore totale nodi
        self.node_list = []      # Elenco semplice degli ID [1, 2, 3...]

    def load_data(self, filename):
        #FASE 2: IL CARBURANTE - Legge il file e riempie le strutture
        print(f"--- Caricamento file: {filename} ---")
        max_id = 0
        
        try:
            with open(filename, 'r', encoding='latin-1') as f:
                # Saltiamo la prima riga (spesso contiene solo totali inutili)
                first_line = f.readline()
                
                for line in f:
                    # Pulizia della riga
                    parts = line.strip().replace(',', ' ').split()
                    if not parts: continue
                    
                    # Riconoscimento: È una definizione di URL?
                    if len(parts) >= 2 and 'http' in parts[1]:
                        nid = int(parts[0])
                        url = parts[1]
                        self.urls[nid] = url
                        max_id = max(max_id, nid)
                        
                    # Riconoscimento: È un collegamento (Link)?
                    elif len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                        src = int(parts[0])
                        dst = int(parts[1])
                        max_id = max(max_id, src, dst)
                        
                        # Creiamo la lista se è la prima volta che vediamo 'src'
                        if src not in self.links: 
                            self.links[src] = []
                        self.links[src].append(dst)
                        
        except FileNotFoundError:
            print(f"ERRORE CRITICO: Non trovo il file '{filename}'.")
            return False

        # Finalizzazione caricamento
        self.num_nodes = max_id
        self.node_list = range(1, self.num_nodes + 1) # Creiamo la lista ID da 1 a N
        
        # Calcoliamo i gradi di uscita per TUTTI i nodi
        for i in self.node_list:
            if i in self.links:
                self.out_degree[i] = len(self.links[i])
            else:
                # Se non è in self.links, è un DANGLING NODE (Vicolo cieco)
                self.out_degree[i] = 0 
                self.links[i] = []
                
        print(f"Grafo pronto: {self.num_nodes} nodi.")
        return True

    def compute_rank(self):
        """FASE 3: IL MOTORE - L'algoritmo matematico"""
        print("\n--- Inizio Calcolo PageRank ---")
        start_time = time.time()
        
        N = self.num_nodes
        
        # Stato Iniziale: Democrazia (1/N a tutti)
        scores = {node: 1.0 / N for node in self.node_list}
        
        # Inizia il ciclo iterativo (Giorno dopo giorno...)
        for it in range(self.max_iter):
            new_scores = {node: 0.0 for node in self.node_list}
            
            # --- GESTIONE DANGLING NODES (Teoria Langville & Meyer) ---
            # 1. Calcoliamo quanti voti sono finiti in vicoli ciechi
            dangling_sum = 0.0
            for node in self.node_list:
                if self.out_degree[node] == 0:
                    dangling_sum += scores[node]
            
            # 2. Calcoliamo la "Paga Base" universale
            # (Tassa Teletrasporto + Voti Dangling Reciclati) / N
            base_val = (self.m + (self.d * dangling_sum)) / N
            
            # Diamo la paga base a tutti
            for node in self.node_list:
                new_scores[node] = base_val
                
            # --- FASE DI SPINTA (PUSH) ---
            # Chi ha link, distribuisce il proprio voto (tassato all'85%)
            for src in self.node_list:
                n_out = self.out_degree[src]
                if n_out > 0:
                    # Calcola la fetta per ogni amico
                    share = (scores[src] * self.d) / n_out
                    # Spedisci la fetta
                    for dst in self.links[src]:
                        if dst <= N: # Controllo di sicurezza
                            new_scores[dst] += share
                            
            # Controllo Convergenza: I punteggi sono cambiati rispetto a ieri?
            diff = sum(abs(new_scores[n] - scores[n]) for n in self.node_list)
            scores = new_scores # Aggiorniamo per il domani
            
            if diff < self.tol:
                print(f"Il sistema si è stabilizzato dopo {it+1} iterazioni.")
                break
            
        end_time = time.time()
        print(f"Tempo di calcolo: {end_time - start_time:.4f} secondi")
        return scores

    def print_top_n(self, scores, n=10):
        """FASE 4: IL CRUSCOTTO - Mostra i vincitori"""
        # Ordiniamo il dizionario dei punteggi
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n=== CLASSIFICA FINALE (TOP {n}) ===")
        print(f"{'Pos':<4} | {'Score':<10} | {'ID':<5} | {'URL'}")
        print("-" * 80)
        
        for i in range(min(n, len(ranked))):
            node_id, score = ranked[i]
            url = self.urls.get(node_id, "N/A") # Recuperiamo il nome leggibile
            
            # Tagliamo URL troppo lunghi per bellezza
            if len(url) > 50: url = url[:47] + "..."
            
            print(f"{i+1:<4} | {score:.6f}   | {node_id:<5} | {url}")

# --- AVVIO ---
if __name__ == "__main__":
    # Creiamo il motore
    engine = PageRankEngine(damping_factor=0.85)
    
    # Carichiamo il file (Assicurati che il nome sia corretto!)
    filename = 'hollins (2).dat' 
    
    if engine.load_data(filename):
        # Calcoliamo
        final_scores = engine.compute_rank()
        # Stampiamo
        engine.print_top_n(final_scores, 10)