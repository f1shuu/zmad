"""
MNIST_cli.py
Klasyfikator cyfr 0–9 dla zbioru MNIST z interfejsem tekstowym.

Etapy:
1. Pobranie danych MNIST z mirrora GitHub (4 pliki .gz w formacie IDX).
2. Wczytanie danych do tablic NumPy.
3. Normalizacja pikseli do przedziału [0, 1].
4. Trening MLPClassifier (scikit-learn).
5. Interfejs tekstowy:
   - [1] pokaż przykładowe cyfry z MNIST z predykcjami,
   - [2] rozpoznaj cyfrę z własnego pliku graficznego,
   - [3] zakończ program.
"""

import os
import gzip
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# 1. Pobieranie i wczytywanie MNIST (pliki IDX .gz z mirrora)
# ============================================================
# Lustro plików MNIST (kopie 1:1 z oryginału)
# Każda pozycja: (URL, nazwa_pliku_na_dysku)
FILES = {
    "train_images": (
        "https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz",
        "train-images-idx3-ubyte.gz"
    ),
    "train_labels": (
        "https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz",
        "train-labels-idx1-ubyte.gz"
    ),
    "test_images": (
        "https://raw.githubusercontent.com/fgnt/mnist/master/t10k-images-idx3-ubyte.gz",
        "t10k-images-idx3-ubyte.gz"
    ),
    "test_labels": (
        "https://raw.githubusercontent.com/fgnt/mnist/master/t10k-labels-idx1-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    )
}

def download(url: str, filename: str) -> None:
    """
    Pobiera plik z podanego URL, jeśli:
    - plik nie istnieje, albo
    - istnieje, ale ma rozmiar 0 bajtów (np. poprzednie pobranie się przerwało).
    """
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        print(f"Plik już istnieje: {filename} (rozmiar: {os.path.getsize(filename)} B)")
        return

    print(f"Pobieram {url} -> {filename}")
    try:
        urllib.request.urlretrieve(url, filename)
    except Exception as e:
        print(f"Błąd podczas pobierania {url}: {e}")
        raise

def load_mnist(data_dir: str = "mnist_data"):
    """
    Pobiera (jeśli trzeba) i wczytuje dane MNIST z plików IDX (.gz).

    Zwraca:
        (X_train, y_train), (X_test, y_test)

    gdzie:
        X_train, X_test: ndarray o kształcie [n_próbek, 784]
        y_train, y_test: etykiety (0–9) jako ndarray [n_próbek]
    """
    os.makedirs(data_dir, exist_ok=True)

    # 1. Pobranie plików (jeśli trzeba)
    for key, (url, fname) in FILES.items():
        path = os.path.join(data_dir, fname)
        download(url, path)

    # 2. Funkcje pomocnicze do odczytu formatu IDX (obrazy i etykiety)
    def read_images(path: str) -> np.ndarray:
        """
        Czyta plik z obrazami w formacie IDX (train-images lub t10k-images).
        Zwraca tablicę [n_images, 784] typu uint8.
        """
        with gzip.open(path, "rb") as f:
            magic = int.from_bytes(f.read(4), "big")
            if magic != 2051:
                raise ValueError(f"Zły magic number w {path}: {magic}")

            n_images = int.from_bytes(f.read(4), "big")
            n_rows = int.from_bytes(f.read(4), "big")
            n_cols = int.from_bytes(f.read(4), "big")

            data = f.read(n_images * n_rows * n_cols)
            images = np.frombuffer(data, dtype=np.uint8)
            images = images.reshape(n_images, n_rows * n_cols)
            return images

    def read_labels(path: str) -> np.ndarray:
        """
        Czyta plik z etykietami w formacie IDX (train-labels lub t10k-labels).
        Zwraca tablicę [n_labels] typu uint8.
        """
        with gzip.open(path, "rb") as f:
            magic = int.from_bytes(f.read(4), "big")
            if magic != 2049:
                raise ValueError(f"Zły magic number w {path}: {magic}")

            n_labels = int.from_bytes(f.read(4), "big")
            data = f.read(n_labels)
            labels = np.frombuffer(data, dtype=np.uint8)
            return labels

    # Ścieżki do konkretnych plików
    train_images_path = os.path.join(data_dir, FILES["train_images"][1])
    train_labels_path = os.path.join(data_dir, FILES["train_labels"][1])
    test_images_path = os.path.join(data_dir, FILES["test_images"][1])
    test_labels_path = os.path.join(data_dir, FILES["test_labels"][1])

    # Faktyczne wczytanie danych
    X_train = read_images(train_images_path)
    y_train = read_labels(train_labels_path)
    X_test = read_images(test_images_path)
    y_test = read_labels(test_labels_path)

    return (X_train, y_train), (X_test, y_test)

# ============================================================
# 2. Funkcje pomocnicze: wizualizacja i predykcja dla własnego obrazka
# ============================================================
def show_examples_from_test(X: np.ndarray,
                            y_true: np.ndarray,
                            model: MLPClassifier,
                            n_samples: int = 9) -> None:
    """
    Wyświetla kilka losowych przykładów z MNIST (zbiór testowy)
    wraz z predykcjami modelu.
    """
    indices = np.random.choice(len(X), n_samples, replace=False)

    plt.figure(figsize=(8, 8))
    for i, idx in enumerate(indices, start=1):
        img = X[idx].reshape(28, 28)
        true_label = int(y_true[idx])

        # Predykcja modelu dla pojedynczego przykładu
        x_single = X[idx].reshape(1, -1)
        pred_label = int(model.predict(x_single)[0])

        plt.subplot(3, 3, i)
        plt.imshow(img, cmap="gray")
        plt.title(f"Prawda: {true_label}\nPred: {pred_label}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def predict_custom_image(model: MLPClassifier, image_path: str):
    """
    Wczytuje własny rysunek cyfry, przygotowuje go "w stylu MNIST"
    (przycinanie do cyfry, centrowanie w 28x28) i zwraca
    (pred_label, proba, canvas_28x28).
    """
    # 1. Wczytanie obrazu i konwersja do skali szarości
    img = Image.open(image_path).convert("L")  # grayscale

    # 2. Zamiana na tablicę, odwrócenie kolorów
    arr = np.array(img)               # [H, W], 0..255, cyfra CZARNA na białym tle
    arr = 255 - arr                   # cyfra JASNA, tło ciemne (jak w MNIST)

    # 3. Progowanie, żeby pozbyć się bardzo jasnego szumu
    #    (80 możesz potem dostroić; im wyżej, tym cieńsza cyfra)
    thresh = 80
    bin_arr = (arr > thresh).astype(np.uint8)  # 1 = cyfra, 0 = tło

    # 4. Szukamy bounding box cyfry
    coords = np.argwhere(bin_arr == 1)
    if coords.size == 0:
        raise ValueError("Nie znalazłem żadnych niezerowych pikseli (cyfra nie narysowana?)")

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1   # +1 bo górna granica jest wyłączna

    # Przycięcie do cyfry
    digit = arr[y0:y1, x0:x1]         # to jest "rdzeń" cyfry

    # 5. Przeskalowanie rdzenia do maks. 20x20 z zachowaniem proporcji
    digit_img = Image.fromarray(digit.astype(np.uint8))
    digit_img.thumbnail((20, 20), Image.Resampling.LANCZOS)
    digit_resized = np.array(digit_img)  # [h, w]

    h, w = digit_resized.shape

    # 6. Płótno 28x28 (jak MNIST) i centrowanie
    canvas = np.zeros((28, 28), dtype=np.float32)
    y_off = (28 - h) // 2
    x_off = (28 - w) // 2
    canvas[y_off:y_off + h, x_off:x_off + w] = digit_resized.astype(np.float32)

    # 7. Normalizacja do [0, 1]
    canvas = canvas / 255.0

    # 8. Predykcja
    x_flat = canvas.reshape(1, -1)
    pred_label = int(model.predict(x_flat)[0])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_flat)[0]
    else:
        proba = None

    return pred_label, proba, canvas

# ============================================================
# 3. Główna logika: trening + interfejs tekstowy
# ============================================================
def main():
    # 1. Wczytanie danych
    print("Wczytywanie danych MNIST (z mirrora GitHub)...")
    (X_train, y_train), (X_test, y_test) = load_mnist(data_dir="mnist_data")

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    # X_*: [n_próbek, 784], y_*: [n_próbek]

    # 2. Normalizacja pikseli do [0, 1]
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # 3. Definicja perceptronu wielowarstwowego (MLP)
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),   # dwie warstwy ukryte
        activation="relu",              # funkcja aktywacji ReLU
        solver="adam",                  # optymalizator Adam
        batch_size=128,
        learning_rate_init=0.001,
        max_iter=20,                    # możesz zwiększyć, jeśli chcesz lepszą dokładność
        random_state=42,
        verbose=True,                   # wypisywanie postępu uczenia
    )

    # 4. Trening
    print("\n== Trening MLP na zbiorze treningowym ==")
    mlp.fit(X_train, y_train)

    # 5. Ewaluacja na zbiorze testowym
    print("\n== Ewaluacja na zbiorze testowym ==")
    y_pred = mlp.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy (dokładność) na zbiorze testowym: {acc:.4f}\n")

    print("Raport klasyfikacji:")
    print(classification_report(y_test, y_pred))

    print("Macierz pomyłek:")
    print(confusion_matrix(y_test, y_pred))

    # 6. Prosty interfejs tekstowy
    while True:
        print("\n=======================================")
        print("MENU:")
        print("[1] Pokaż przykładowe cyfry z MNIST z predykcjami")
        print("[2] Rozpoznaj cyfrę z własnego pliku graficznego")
        print("[3] Zakończ program")
        print("=======================================")

        choice = input("Wybierz opcję (1/2/3): ").strip()

        if choice == "1":
            # Przykłady ze zbioru testowego
            try:
                n_samples_str = input("Ile przykładów pokazać? [domyślnie 9]: ").strip()
                n_samples = int(n_samples_str) if n_samples_str else 9
            except ValueError:
                n_samples = 9


            print(f"Wyświetlam {n_samples} losowych przykładów z MNIST...")
            show_examples_from_test(X_test, y_test, mlp, n_samples=n_samples)

        elif choice == "2":
            # Rozpoznawanie własnego obrazka
            image_path = input(
                "Podaj pełną ścieżkę do pliku z obrazem (np. C:\\...\\moja_cyfra.png):\n> "
            ).strip()

            if not os.path.isfile(image_path):
                print("Plik nie istnieje. Sprawdź ścieżkę i spróbuj ponownie.")
                continue

            try:
                pred_label, proba, arr = predict_custom_image(mlp, image_path)
                print(f"\nSieć przewiduje cyfrę: {pred_label}")

                if proba is not None:
                    print("Prawdopodobieństwa klas (0..9):")
                    for digit, p in enumerate(proba):
                        print(f"  {digit}: {p:.3f}")

                # Wyświetlenie obrazu w formie 28x28 (po przetworzeniu)
                plt.figure()
                plt.imshow(arr, cmap="gray")
                plt.title(f"Twoja cyfra – przewidywana: {pred_label}")
                plt.axis("off")
                plt.show()

            except Exception as e:
                print(f"Wystąpił błąd podczas przetwarzania obrazu: {e}")

        elif choice == "3":
            print("Zamykanie programu.")
            break

        else:
            print("Nieznana opcja. Wpisz 1, 2 lub 3.")

if __name__ == "__main__":
    main()
