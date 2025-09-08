
---

# Optical Flow Pose Tracking

![ROS](https://img.shields.io/badge/ROS-Noetic-blue) ![C++](https://img.shields.io/badge/C++-14-blue) ![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green)

## Descrizione

Questo progetto implementa un sistema di **tracciamento di punti chiave e recupero della posa della camera** utilizzando **ROS** e **OpenCV**.

Il sistema segue i punti tra frame consecutivi usando **optical flow**, distinguendo tra punti **dinamici** e **statici**, con supporto per immagini **RGB e depth**.

Principali funzionalità:

* Estrazione di feature in regioni di interesse (ROI) definite.
* Tracciamento di punti 2D/3D nel tempo tramite Lucas-Kanade Optical Flow.
* Identificazione di punti dinamici oltre una soglia di movimento configurabile.
* Visualizzazione dei punti tracciati in tempo reale.
* Reinizializzazione automatica dei punti persi.
* Pubblicazione dei frame elaborati su topic ROS (`/optical_flow/output_video`).

---

## Struttura dei file

```
├── src/
│   ├── OpticalFlow.h           # Definizione classe OpticalFlow
│   ├── OpticalFlow.cpp         # Implementazione OpticalFlow
│   ├── OpticalFlowPose.h       # Definizione classe OpticalFlowPose
│   ├── OpticalFlowPose.cpp     # Implementazione OpticalFlowPose
│   ├── TrackedMatch.h          # Struct/class per punti tracciati 2D/3D
│   ├── Visualizer.h            # Funzioni opzionali di visualizzazione
│   └── data.h                  # Definizioni costanti e helper
├── launch/
│   └── optical_flow.launch     # File di launch ROS
├── CMakeLists.txt
└── package.xml
```

---

## Requisiti

* ROS Noetic
* OpenCV ≥ 4.0
* `cv_bridge`, `image_transport`, `message_filters`
* C++14 o superiore

---

## Installazione

Clona il repository nel workspace ROS:

```bash
cd ~/catkin_ws/src
git clone <REPO_URL> optical_flow_pose_tracking
cd ..
catkin_make
source devel/setup.bash
```

Installa le dipendenze ROS:

```bash
sudo apt-get install ros-noetic-cv-bridge ros-noetic-image-transport ros-noetic-message-filters
```

---

## Utilizzo

1. Avvia il nodo ROS della camera RGB-D o utilizza un dataset simulato.
2. Lancia il nodo di optical flow:

```bash
roslaunch optical_flow_pose_tracking optical_flow.launch
```

3. Visualizza le immagini elaborate con:

```bash
rosrun rqt_image_view rqt_image_view
```

**Colori dei punti:**

* **Verde**: punti dinamici
* **Blu**: punti statici

---

## Parametri configurabili

| Parametro           | Default | Descrizione                                            |
| ------------------- | ------- | ------------------------------------------------------ |
| `threshold`         | 10.0    | Soglia di movimento per considerare un punto dinamico. |
| `/video1/image_raw` | N/A     | Topic ROS per le immagini RGB.                         |
| `/video1/depth`     | N/A     | Topic ROS per le immagini depth.                       |

---

## TrackedMatch

`TrackedMatch` è la **struttura dati principale** per rappresentare ogni punto tracciato tra i frame.
Permette di mantenere la storia di ciascun punto e di distinguere tra punti **statici** e **dinamici**.

**Campi principali:**

* `cv::Point2f pt` → coordinate 2D nel frame corrente.
* `cv::Point3f position_3d` → coordinate 3D usando la depth map.
* `bool is_active` → indica se il punto è ancora tracciabile.
* `bool dynamic_point` → indica se il punto è dinamico (movimento > soglia).
* `std::vector<cv::Point3f> history` → cronologia delle posizioni 3D del punto.

**Flow di esecuzione:**

1. **Inizializzazione**: i punti feature vengono selezionati nella ROI e salvati come `TrackedMatch`.
2. **Tracciamento**: per ogni frame, Lucas-Kanade aggiorna posizione e stato del punto.
3. **Visualizzazione**: punti dinamici/attivi vengono mostrati sull’immagine.
4. **Reinizializzazione**: se tutti i punti vengono persi, nuovi punti vengono creati.

**Diagramma mentale semplificato:**

```
[Primo frame] --> estrazione feature --> crea TrackedMatch
     |
[Frame successivi] --> LK Optical Flow --> aggiorna pt, position_3d, dynamic_point
     |
Recupero posa + visualizzazione --> Publish su ROS topic
     |
Se punti persi --> re-inizializzazione
```

---

## Output / Visualizzazione

* Frame pubblicato su `/optical_flow/output_video`.
* Visualizzabile con `rqt_image_view`.
* Punti dinamici: **verde**
* Punti statici: **blu**

---

## Note aggiuntive

* Supporta ROI personalizzate per ottimizzare il calcolo delle feature.
* La funzione `recoverPose` può essere estesa per calcolo reale della posa (Essential Matrix o PnP).
* La soglia di movimento (`threshold`) può essere regolata in base alla scena.
* I punti dinamici rimangono marcati fino a reinizializzazione.


