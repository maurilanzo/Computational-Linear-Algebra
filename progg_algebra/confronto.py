import networkx as nx
from metodi import PageRankEngine # Importiamo la tua classe dal file precedente

def compare_methods(filename):
    print(f"--- CONFRONTO MANUALE vs NETWORKX ({filename}) ---")
    
    # 1. CALCOLO MANUALE (Il tuo codice)
    print("1. Esecuzione PageRank Manuale...")
    my_engine = PageRankEngine(damping_factor=0.85, tolerance=1e-9, max_iter=100)
    if not my_engine.load_data(filename): return
    my_scores = my_engine.compute_rank()
    
    # 2. CALCOLO NETWORKX (La libreria)
    print("2. Esecuzione NetworkX...")
    # Creiamo il grafo per NetworkX
    G = nx.DiGraph()
    
    # Aggiungiamo i nodi (anche quelli isolati/dangling che il tuo load_data trova)
    # Nota: dobbiamo replicare la logica di caricamento per essere fedeli
    with open(filename, 'r', encoding='latin-1') as f:
        f.readline()
        for line in f:
            parts = line.strip().replace(',', ' ').split()
            if not parts: continue
            
            if len(parts) >= 2 and 'http' in parts[1]:
                nid = int(parts[0])
                G.add_node(nid) # Aggiunge nodo (anche se isolato)
            elif len(parts) == 2 and parts[0].isdigit():
                src, dst = int(parts[0]), int(parts[1])
                G.add_edge(src, dst)

    # Calcolo automatico
    # NetworkX gestisce i dangling nodes automaticamente con la stessa logica (personalization vector uniforme)
    nx_scores = nx.pagerank(G, alpha=0.85, tol=1e-9)
    
    # 3. CONFRONTO
    print("\n--- RISULTATI A CONFRONTO (Top 5) ---")
    print(f"{'ID':<5} | {'Mio Score':<12} | {'NetworkX':<12} | {'Diff':<12}")
    print("-" * 50)
    
    # Ordiniamo per il mio score
    ranked_ids = sorted(my_scores.keys(), key=lambda k: my_scores[k], reverse=True)[:5]
    
    max_diff = 0.0
    
    for nid in ranked_ids:
        s1 = my_scores.get(nid, 0)
        s2 = nx_scores.get(nid, 0)
        diff = abs(s1 - s2)
        max_diff = max(max_diff, diff)
        
        print(f"{nid:<5} | {s1:.9f}    | {s2:.9f}    | {diff:.2e}")

    print("-" * 50)
    print(f"Differenza Massima Assoluta: {max_diff:.2e}")
    
    if max_diff < 1e-6:
        print("✅ SUCCESSO: I risultati coincidono!")
    else:
        print("⚠️ ATTENZIONE: C'è una discrepanza significativa.")

# --- AVVIO ---
if __name__ == "__main__":
    compare_methods('hollins (2).dat')