# import re
# import matplotlib.pyplot as plt

# def leggi_file(nome_file):
#     dati = {}
#     with open(nome_file, "r") as f:
#         for linea in f:
#             # Trova PointID
#             match_id = re.search(r"PointID:(\d+)", linea)
#             if not match_id:
#                 continue
#             point_id = int(match_id.group(1))

#             # Trova tutte le triplette
#             triplette = re.findall(r"\(([^)]+)\)", linea)
#             valori = []
#             for t in triplette:
#                 x, y, z = map(float, t.split(","))
#                 valori.append((x, y, z))

#             dati[point_id] = valori
#     return dati


# def plotta_dati(dati, max_points=None):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection="3d")

#     # ordino per PointID e prendo solo i primi max_points
#     for pid, triplette in sorted(dati.items())[:max_points]:
#         xs, ys, zs = zip(*triplette)
#         ax.plot(xs, ys, zs, marker="o", linestyle="-", label=f"PointID {pid}")

#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")
#     ax.legend()
#     plt.show()



# if __name__ == "__main__":
#     nome_file = "/home/corbe/heart_ws/src/heart_pkg/scripts/tracked_features.txt"  # cambia con il tuo file
#     dati = leggi_file(nome_file)
#     plotta_dati(dati, max_points=5)


# import re
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# def leggi_primi_punti(fname):
#     dati = {}
#     with open(fname) as f:
#         for linea in f:
#             match = re.search(r"PointID:(\d+).*?History:\(([^)]+)\)", linea)
#             if match:
#                 pid = int(match.group(1))
#                 primo = match.group(2)
#                 x, y, z = map(float, primo.split(","))
#                 dati[pid] = (x, y, z)
#     return dati

# def fit_plane_least_squares(points):
#     xs, ys, zs = zip(*points)
#     A = np.vstack([xs, ys, np.ones_like(xs)]).T
#     return np.linalg.lstsq(A, zs, rcond=None)[0]  # (a, b, c)

# def plot_plane(a, b, c, points):
#     xs, ys, zs = zip(*points)
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(xs, ys, zs, color='b', label='Punti')

#     xx, yy = np.meshgrid(np.linspace(min(xs), max(xs), 10),
#                          np.linspace(min(ys), max(ys), 10))
#     zz = a*xx + b*yy + c
#     ax.plot_surface(xx, yy, zz, alpha=0.5, color='r')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     plt.legend()
#     plt.show()


# if __name__ == "__main__":
#     dati = leggi_primi_punti("/home/corbe/heart_ws/src/heart_pkg/scripts/tracked_features.txt")
#     primi = list(dati.values())
#     a, b, c = fit_plane_least_squares(primi)
#     print(f"Piano fit: z = {a:.4f} x + {b:.4f} y + {c:.4f}")
#     plot_plane(a, b, c, primi)


import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # solo per possibile debug 3D

def parse_file(fname):
    """
    Legge il file e costruisce un dizionario:
      data[point_id] = [(x,y,z), (x,y,z), ...]   # ordine preservato (history)
    """
    data = {}
    with open(fname, 'r') as f:
        content = f.read()

    # Split in blocchi che iniziano con "PointID:"
    blocks = re.split(r'(?=PointID:\d+)', content)
    for block in blocks:
        if not block.strip():
            continue
        m = re.match(r'PointID:(\d+).*?History:((?:\([^)]+\))+)', block, flags=re.S)
        if not m:
            continue
        pid = int(m.group(1))
        history_str = m.group(2)

        # Trova tutte le triplette "(x,y,z)"
        triplets = re.findall(r'\(([^)]+)\)', history_str)
        coords = []
        for t in triplets:
            # alcuni valori potrebbero avere spazi; split sicuro
            parts = [p.strip() for p in t.split(',')]
            if len(parts) != 3:
                continue
            x, y, z = map(float, parts)
            coords.append((x, y, z))
        if coords:
            data[pid] = coords
    return data

def plane_from_first_positions(data):
    """
    Calcola piano best-fit usando le prime posizioni di ogni PointID.
    Restituisce:
      centroid: punto sul piano (media dei primi punti)
      normal: vettore normale unitario (lunghezza 1)
    Metodo: SVD su punti centrati (PCA).
    """
    first_points = []
    ids = []
    for pid, hist in sorted(data.items()):
        if len(hist) >= 1:
            first_points.append(hist[0])
            ids.append(pid)
    P = np.array(first_points)  # shape (N,3)
    centroid = P.mean(axis=0)
    # SVD su dati centrati
    U, S, Vt = np.linalg.svd(P - centroid)
    normal = Vt[-1, :]  # ultimo vettore: direzione di min varianza
    # normalizzare
    normal = normal / np.linalg.norm(normal)
    return centroid, normal, ids

def signed_distances_along_normal(data, plane_point, normal):
    """
    Per ogni pointID produce un array di distanze firmate:
      d = n · (p - plane_point)
    Ritorna dict: distances[pid] = np.array([...])
    """
    distances = {}
    for pid, hist in sorted(data.items()):
        pts = np.array(hist)  # (M,3)
        vecs = pts - plane_point  # (M,3)
        ds = vecs.dot(normal)    # proiezione scalare su normal (signed)
        distances[pid] = ds
    return distances

def plot_distances(distances, max_points=None, figsize=(10,6)):
    """
    Disegna le serie di distanze nel tempo (indice history come 'time').
    max_points: se specificato, plotta solo i primi max_points PointID.
    """
    plt.figure(figsize=figsize)
    items = sorted(distances.items())
    if max_points is not None:
        items = items[:max_points]
    for pid, ds in items:
        t = np.arange(len(ds))  # tempo artificiale: indice nella history
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
    data = parse_file(fname)
    if not data:
        raise RuntimeError("Nessun dato trovato nel file.")
    plane_point, normal, ids = plane_from_first_positions(data)
    print("Centroid (punto sul piano):", plane_point)
    print("Normal unitario:", normal)
    distances = signed_distances_along_normal(data, plane_point, normal)
    plot_distances(distances, max_points=max_to_plot)
    return data, plane_point, normal, distances

if __name__ == "__main__":
    # Cambia il nome del file con il tuo
    filename = "/home/corbe/heart_ws/src/heart_pkg/scripts/tracked_features2.txt"
    # Se vuoi plottare solo i primi 5 PointID metti max_to_plot=5
    data, plane_point, normal, distances = example_usage(filename, max_to_plot=10)
