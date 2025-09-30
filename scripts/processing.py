import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # solo necessario per debug 3D

def parse_file(fname):
    """
    Legge un file di tracked features e costruisce un dizionario.
    
    Args:
        fname (str): percorso del file di testo.
        
    Returns:
        dict: data[point_id] = [(x,y,z), (x,y,z), ...] contenente
              la storia completa dei punti.
    """
    data = {}
    with open(fname, 'r') as f:
        content = f.read()

    # Split del contenuto in blocchi che iniziano con "PointID:"
    blocks = re.split(r'(?=PointID:\d+)', content)
    for block in blocks:
        if not block.strip():  # salto blocchi vuoti
            continue
        # Regex per catturare PointID e la History associata
        m = re.match(r'PointID:(\d+).*?History:((?:\([^)]+\))+)', block, flags=re.S)
        if not m:
            continue
        pid = int(m.group(1))
        history_str = m.group(2)

        # Trova tutte le triplette "(x,y,z)" nella History
        triplets = re.findall(r'\(([^)]+)\)', history_str)
        coords = []
        for t in triplets:
            parts = [p.strip() for p in t.split(',')]  # rimuove eventuali spazi
            if len(parts) != 3:
                continue
            x, y, z = map(float, parts)
            coords.append((x, y, z))
        if coords:
            data[pid] = coords
    return data


def plane_from_first_positions(data):
    """
    Calcola un piano "best-fit" usando le prime posizioni di ciascun PointID.
    
    Args:
        data (dict): dizionario con tutte le posizioni dei punti.
        
    Returns:
        centroid (np.array): punto medio dei primi punti.
        normal (np.array): vettore normale unitario al piano.
        ids (list): lista dei PointID considerati.
    """
    first_points = []
    ids = []
    for pid, hist in sorted(data.items()):
        if len(hist) >= 1:
            first_points.append(hist[0])
            ids.append(pid)
    P = np.array(first_points)  # matrice Nx3
    centroid = P.mean(axis=0)   # media dei punti: punto sul piano

    # PCA tramite SVD per trovare la direzione di minima varianza (normale al piano)
    U, S, Vt = np.linalg.svd(P - centroid)
    normal = Vt[-1, :]           # ultimo vettore: min varianza
    normal = normal / np.linalg.norm(normal)  # normalizzazione a lunghezza 1
    return centroid, normal, ids


def signed_distances_along_normal(data, plane_point, normal):
    """
    Calcola la distanza firmata di ciascun punto dalla plane lungo la normale.
    
    Args:
        data (dict): tutte le posizioni dei punti.
        plane_point (np.array): punto di riferimento sul piano.
        normal (np.array): normale unitaria al piano.
    
    Returns:
        dict: distances[pid] = np.array([...]) con le distanze firmate per ogni PointID.
    """
    distances = {}
    for pid, hist in sorted(data.items()):
        pts = np.array(hist)        # shape (M,3)
        vecs = pts - plane_point    # vettori dai punti al piano
        ds = vecs.dot(normal)       # proiezione lungo la normale (signed)
        distances[pid] = ds
    return distances


def plot_distances(distances, max_points=None, point_ids=None, figsize=(10,6)):
    """
    Visualizza le distanze firmate nel tempo.

    Args:
        distances (dict): output di signed_distances_along_normal.
        max_points (int): opzionale, plotta solo i primi max_points PointID.
        point_ids (list): opzionale, lista di PointID da plottare.
        figsize (tuple): dimensione della figura.
    """
    plt.figure(figsize=figsize)

    items = sorted(distances.items())

    # Se specificato, filtriamo solo i PointID richiesti
    if point_ids is not None:
        items = [item for item in items if item[0] in point_ids]
    elif max_points is not None:
        items = items[:max_points]

    for pid, ds in items:
        t = np.arange(len(ds))  # indice temporale artificiale
        plt.plot(t, ds, marker='o', linestyle='-', label=f'Point {pid}')

    plt.axhline(0, color='k', linewidth=0.7, linestyle='--')
    plt.xlabel('frame / step nella History (0 = prima posizione)')
    plt.ylabel('distanza firmata lungo la normale al piano')
    plt.title('Movimento dei punti proiettato sulla normale al piano')
    plt.legend(fontsize='small', ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



def example_usage(fname, max_to_plot=None):
    """
    Esegue parsing, calcola piano, distanze e plotta le prime max_to_plot serie.
    
    Args:
        fname (str): file di tracked features.
        max_to_plot (int): opzionale, numero massimo di punti da visualizzare.
    
    Returns:
        data, plane_point, normal, distances
    """
    data = parse_file(fname)
    if not data:
        raise RuntimeError("Nessun dato trovato nel file.")
    plane_point, normal, ids = plane_from_first_positions(data)
    print("Centroid (punto sul piano):", plane_point)
    print("Normal unitario:", normal)
    distances = signed_distances_along_normal(data, plane_point, normal)
    plot_distances(distances, max_points=max_to_plot)
    return data, plane_point, normal, distances

import numpy as np

import numpy as np

def filter_and_rank_distances(distances,
                              min_frames=3,
                              min_variation=0.5,
                              remove_negative_only=True,
                              max_derivative_std=None,
                              top_k=None,
                              verbose=False):
    """
    Filtra i PointID in base a criteri di durata, ampiezza, segno e coerenza,
    e restituisce i punti ordinati per "qualità" (durata * ampiezza / rumore).

    Args:
        distances (dict): output di signed_distances_along_normal.
        min_frames (int): numero minimo di step per considerare il punto valido.
        min_variation (float): variazione minima lungo la normale per essere considerato significativo.
        remove_negative_only (bool): se True, rimuove punti sempre negativi.
        max_derivative_std (float or None): se impostato, rimuove punti con derivata troppo rumorosa.
        top_k (int or None): se specificato, restituisce solo i primi top_k punti migliori.
        verbose (bool): stampa motivi di scarto dei punti.

    Returns:
        dict: dizionario filtrato {pid: ds} ordinato per qualità.
        list: lista di PointID ordinati per qualità decrescente.
    """
    filtered = {}
    scores = {}

    for pid, ds in distances.items():
        ds = np.array(ds)
        reasons = []

        # Durata minima
        if len(ds) < min_frames:
            reasons.append(f"too short ({len(ds)} frames)")

        # Variazione minima
        var = ds.max() - ds.min()
        if var < min_variation:
            reasons.append(f"variation too small ({var:.3f})")

        # Solo valori negativi
        if remove_negative_only and np.all(ds < 0):
            reasons.append("all negative")

        # Rumore della derivata
        deriv_std = np.std(np.diff(ds)) if len(ds) > 1 else 0
        if max_derivative_std is not None and deriv_std > max_derivative_std:
            reasons.append(f"derivative too noisy ({deriv_std:.3f})")

        if not reasons:
            filtered[pid] = ds
            # Calcolo punteggio: più lungo e più ampio, meno rumore
            score = (len(ds) * var) / (deriv_std + 1e-6)
            scores[pid] = score
        elif verbose:
            print(f"Point {pid} filtered out: {', '.join(reasons)}")

    # Ordina PointID per punteggio decrescente
    sorted_pids = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

    # Se richiesto, seleziona solo i top_k
    if top_k is not None:
        top_pids = sorted_pids[:top_k]
        filtered = {pid: filtered[pid] for pid in top_pids}
        sorted_pids = top_pids

    return filtered, sorted_pids


def plot_top_points(distances, sorted_pids, top_n=5, figsize=(10,6)):
    """
    Plot dei top-N punti filtrati ordinati per qualità.

    Args:
        distances (dict): dizionario {pid: ds} di distanze firmate.
        sorted_pids (list): lista di PointID ordinati per punteggio decrescente.
        top_n (int): numero massimo di punti da plottare.
        figsize (tuple): dimensione della figura.
    """
    plt.figure(figsize=figsize)
    count = min(top_n, len(sorted_pids))
    
    for i in range(count):
        pid = sorted_pids[i]
        ds = np.array(distances[pid])
        t = np.arange(len(ds))
        plt.plot(t, ds, marker='o', linestyle='-', label=f'Point {pid}')

    plt.axhline(0, color='k', linewidth=0.7, linestyle='--')
    plt.xlabel('Frame / step nella History (0 = prima posizione)')
    plt.ylabel('Distanza firmata lungo la normale al piano')
    plt.title(f'Top {count} punti filtrati per qualità')
    plt.legend(fontsize='small', ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------- Esempio completo ----------------
if __name__ == "__main__":
    filename = "/home/corbe/heart_ws/src/heart_pkg/positions/tracked_features.txt"

    # Parsing e calcolo distanze
    data, plane_point, normal, distances = example_usage(filename, max_to_plot=None)

    # Filtraggio avanzato e ranking
    distances_filtered, sorted_pids = filter_and_rank_distances(
        distances,
        min_frames=200,
        min_variation=0.5,
        max_derivative_std=0.7,  # esempio soglia rumore
        top_k=20,
        verbose=True
    )

    # Plot dei top punti
    plot_top_points(distances_filtered, sorted_pids, top_n=10)

