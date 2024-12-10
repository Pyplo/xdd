# Autonomous Vehicle Project

### Opis projektu
Projekt pracy inżynierskiej: autonomiczny pojazd wykorzystujący symulator CARLA i sieci neuronowe.

**Cele projektu:**
1. Śledzenie linii i poruszanie się po drodze.
2. Unikanie kolizji.
3. Reakcje na znaki drogowe i sygnalizację świetlną.

### Technologie
- Symulator: CARLA (0.9.15)
- Język: Python
- Biblioteki: TensorFlow, PyTorch, OpenCV, NumPy, Matplotlib
- IDE: PyCharm

---

### Struktura projektu
```
autonomous_vehicle_project/
│
├── data/                   # Dane i ich przetwarzanie
│   ├── raw/                # Surowe dane (zebrane w CARLA)
│   ├── processed/          # Przetworzone dane
│   └── datasets/           # Zestawy treningowe i walidacyjne
│
├── models/                 # Zapisane modele
│   ├── line_tracking/      # Model do śledzenia linii
│   ├── collision_avoidance/ # Model unikania kolizji
│   ├── traffic_reactions/  # Model reakcji na otoczenie
│   └── combined_model/     # Model łączący funkcje
│
├── scripts/                # Skrypty do uczenia i ewaluacji
│   ├── data_processing.py  # Przetwarzanie danych
│   ├── train_line_tracking.py
│   ├── train_collision_avoidance.py
│   ├── train_traffic_reactions.py
│   └── evaluate.py         # Ewaluacja całego systemu
│
├── utils/                  # Funkcje pomocnicze
│   ├── carla_interaction.py # Interakcja ze środowiskiem CARLA
│   ├── logging.py          # Obsługa logów
│   └── visualization.py    # Wizualizacja wyników
│
├── main.py                 # Główny skrypt uruchamiający system
└── README.md               # Opis projektu
```
## Autor
- **Imię i nazwisko:** Patryk Pytel
- **Kontakt:** Patrykp295@gmail.com
