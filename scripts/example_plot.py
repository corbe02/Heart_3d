import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from scipy.signal import butter, filtfilt, detrend, find_peaks
from scipy.fft import rfft, rfftfreq
from sklearn.decomposition import PCA

# --- Variabili Globali ---
# Sostituisci questo con il percorso corretto del tuo file
FILENAME = "/home/corbe/heart_ws/src/heart_pkg/positions/tracked_features4.txt"
FS = 60 # Frequenza di campionamento in Hz

# ----------------------- Parsing file -----------------------
def parse_file(fname):
    data = {}
    with open(fname, 'r') as f:
        content = f.read()

    blocks = re.split(r'(?=PointID:\d+)', content)
    for block in blocks:
        if not block.strip():
            continue
        m = re.match(r'PointID:(\d+).*?History:((?:\([^)]+\))+)', block, flags=re.S)
        if not m:
            continue
        pid = int(m.group(1))
        history_str = m.group(2)
        triplets = re.findall(r'\(([^)]+)\)', history_str)
        coords = []
        for t in triplets:
            parts = [p.strip() for p in t.split(',')]
            if len(parts) != 3:
                continue
            x, y, z = map(float, parts)
            coords.append((x, y, z))
        if coords:
            data[pid] = coords
    
    if data:
        num_points = [len(coords) for coords in data.values()]
        print(f"Numero minimo di punti per feature: {min(num_points)}")
        print(f"Numero massimo di punti per feature: {max(num_points)}")
        print(f"Numero medio di punti per feature: {np.mean(num_points):.2f}")
    return data

# ----------------------- Plot 3D e 2D -----------------------
def plot_points(data, fx, fy, cx, cy):
    # Omissis per brevità
    pass

# ----------------------- FFT -----------------------
def plot_fft(sig, fs, title="Spettro del segnale"):
    N = len(sig)
    freqs = rfftfreq(N, 1/fs)
    spectrum = np.abs(rfft(sig))
    plt.figure()
    plt.plot(freqs, spectrum)
    plt.xlim(0, 5) # Limita la visualizzazione alle frequenze HR
    plt.xlabel("Frequenza [Hz]")
    plt.ylabel("Ampiezza")
    plt.title(title)
    plt.grid()
    plt.show()

# ----------------------- Estrazione segnali (PCA + Filtraggio HR) -----------------------
def extract_signal_pca(coords, fs=60, n_frames=100):
    """
    Estrae un segnale 1D dal movimento 3D usando PCA
    e lo filtra nella banda tipica del battito (0.5 Hz - 4 Hz).
    """
    # 1. PCA per stimare direzione principale
    n = min(len(coords), n_frames)
    data = np.array(coords[:n])
    pca = PCA(n_components=1)
    pca.fit(data)
    dir_vec = pca.components_[0]   # direzione principale (unit vector)

    # 2. Proiezione di tutte le coordinate sulla direzione trovata
    projected = np.array(coords) @ dir_vec

    # 3. Detrend + filtro passa-banda
    projected_detr = detrend(projected)
    
    # **IMPORTANTE: BANDA STRETTA PER LA FREQUENZA CARDIACA**
    # HR tipica 40-240 BPM -> Frequenze 0.67 Hz - 4 Hz
    low, high = 0.5, 4.0 # Hz
    
    nyquist_freq = fs / 2
    
    # Gestione dell'errore (solo se low/high >= nyquist_freq)
    if low >= nyquist_freq or high >= nyquist_freq:
        print(f"[ERRORE] Frequenze di taglio ({low}-{high}) > Nyquist ({nyquist_freq}). Rivedi FS.")
        return projected_detr, dir_vec

    try:
        b, a = butter(3, [low/nyquist_freq, high/nyquist_freq], btype="band")
        filtered = filtfilt(b, a, projected_detr)
    except ValueError as e:
        print(f"[ERRORE filtro] {e}. Restituisco solo segnale detrend.")
        filtered = projected_detr

    # print(f"[DEBUG] PCA signal range: min={filtered.min():.4f}, max={filtered.max():.4f}")
    return filtered, dir_vec

# ----------------------- Calcolo HR (Basato su FFT) -----------------------
def calculate_hr_fft(sig, fs):
    """
    Calcola l'HR trovando la frequenza con la massima ampiezza nello spettro FFT
    nell'intervallo fisiologico (0.6 Hz a 4.0 Hz).
    """
    N = len(sig)
    freqs = rfftfreq(N, 1/fs)
    spectrum = np.abs(rfft(sig))
    
    # Identifica l'indice per le frequenze fisiologiche HR (0.6 Hz - 4 Hz)
    min_idx = np.where(freqs >= 0.6)[0][0]
    max_idx = np.where(freqs <= 4.0)[0][-1]

    # Trova il picco di ampiezza in quell'intervallo
    if min_idx >= max_idx:
         return 0, 0, 0 # Nessun segnale significativo trovato

    relevant_spectrum = spectrum[min_idx:max_idx]
    
    # Trova l'indice della frequenza dominante
    peak_idx = np.argmax(relevant_spectrum) + min_idx
    
    # Frequenza dominante (Hz)
    dominant_freq = freqs[peak_idx]
    
    # Conversione in Battiti al Minuto (BPM)
    hr_bpm = dominant_freq * 60
    
    return hr_bpm, dominant_freq, spectrum

# ----------------------- PCA per direzione principale -----------------------
def principal_direction(coords, n_frames=100):
    n = min(len(coords), n_frames)
    data = np.array(coords[:n])
    pca = PCA(n_components=1)
    pca.fit(data)
    return pca.components_[0]

def project_on_direction(coords, direction):
    data = np.array(coords)
    return data @ direction

def project_all_features(data, n_frames=100):
    # Omissis per brevità
    pass

# ----------------------- Plot segnali proiettati -----------------------
def plot_projected(projected, fs=60, first_n_seconds=None):
    # Omissis per brevità
    pass

# ----------------------- MAIN -----------------------
if __name__ == "__main__":
    data = parse_file(FILENAME)
    
    if not data:
        print("Nessun dato valido trovato nel file. Uscita.")
        exit()

    # --- Estrazione, Filtraggio e Calcolo HR ---
    first_pid = list(data.keys())[0]
    coords = data[first_pid]

    print(f"\n--- Analisi Segnale PointID {first_pid} (FS={FS} Hz) ---")

    # 1. Estrazione del segnale (PCA + Filtro HR 0.5-4 Hz)
    sig_filtrato, dir_vec = extract_signal_pca(coords, fs=FS, n_frames=100)
    
    # 2. Calcolo HR tramite FFT
    hr_bpm, hr_freq, spectrum = calculate_hr_fft(sig_filtrato, FS)
    
    print(f"Frequenza Cardiaca (HR) stimata: {hr_bpm:.2f} BPM (corrisponde a {hr_freq:.2f} Hz)")
    print("-" * 40)

    # 3. Plot FFT del segnale filtrato (per conferma)
    plot_fft(sig_filtrato, FS, title=f"Spettro Segnale Filtrato (HR Stimata: {hr_bpm:.2f} BPM)")
    
    # 4. Plot Segnale Filtrato nel Tempo
    t = np.arange(len(sig_filtrato)) / FS
    plt.figure(figsize=(10, 4))
    plt.plot(t, sig_filtrato, label=f"Point {first_pid} (Filtrato)")
    plt.xlabel("Tempo [s]")
    plt.ylabel("Segnale Filtrato (Amplitudine)")
    plt.title(f"Segnale proiettato filtrato (HR: {hr_bpm:.2f} BPM)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # --- Plot originali (mantengo l'originale per debug/confronto) ---
    # Controllo FFT su componente Z originale (per vedere rumore)
    z = np.array([c[2] for c in coords])
    plot_fft(z, FS, title="Spettro Componente Z Originale")