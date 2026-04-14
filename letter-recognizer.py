import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.neural_network import MLPClassifier

def preprocess_image_to_canvas28(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("L")
    arr = np.array(img)
    arr = 255 - arr
    thresh = 80
    bin_arr = (arr > thresh).astype(np.uint8)

    coords = np.argwhere(bin_arr == 1)
    if coords.size == 0:
        raise ValueError(f"No white pixels found in {image_path}")

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    digit = arr[y0:y1, x0:x1]

    digit_img = Image.fromarray(digit.astype(np.uint8))
    digit_img.thumbnail((20, 20), Image.Resampling.LANCZOS)
    digit_resized = np.array(digit_img)

    h, w = digit_resized.shape
    canvas = np.zeros((28, 28), dtype=np.float32)
    y_off = (28 - h) // 2
    x_off = (28 - w) // 2
    canvas[y_off:y_off + h, x_off:x_off + w] = digit_resized.astype(np.float32)

    canvas /= 255.0
    return canvas

LETTER_LABEL_MAP = {
    "a": "A", "b": "B", "c": "C", "d": "D", "e": "E", "f": "F",
    "g": "G", "h": "H", "i": "I", "j": "J", "k": "K", "l": "L",
    "m": "M", "n": "N", "o": "O", "p": "P", "q": "Q", "r": "R",
    "s": "S", "t": "T", "u": "U", "v": "V", "w": "W", "x": "X",
    "y": "Y", "z": "Z"
}

def load_custom_data(base_dir: str, label_map: dict):
    X_list = []
    y_list = []

    if not os.path.isdir(base_dir):
        print(f"(Informacja) Brak katalogu {base_dir} – pomijam wczytywanie.")
        return np.empty((0, 784), dtype=np.float32), np.empty((0,), dtype=object)

    for dirname, label in label_map.items():
        folder = os.path.join(base_dir, dirname)
        if not os.path.isdir(folder):
            continue

        pattern = os.path.join(folder, "*.*")
        for path in glob.glob(pattern):
            if not path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue
            try:
                canvas = preprocess_image_to_canvas28(path)
                X_list.append(canvas.reshape(-1))
                y_list.append(label)
            except Exception as e:
                print(f"Pomijam {path}: {e}")

    if not X_list:
        print(f"(Informacja) Nie znaleziono żadnych obrazów w {base_dir}")
        return np.empty((0, 784), dtype=np.float32), np.empty((0,), dtype=object)

    X_custom = np.stack(X_list).astype(np.float32)
    y_custom = np.array(y_list)
    print(f"Wczytano {len(X_custom)} przykładów z {base_dir}")
    return X_custom, y_custom

def show_examples_from_test(X: np.ndarray, y_true: np.ndarray, model: MLPClassifier, n_samples: int = 9):
    if X.size == 0:
        print("Brak danych testowych w tym trybie.")
        return

    n_samples = min(n_samples, len(X))
    indices = np.random.choice(len(X), n_samples, replace=False)

    plt.figure(figsize=(8, 8))
    for i, idx in enumerate(indices, start=1):
        img = X[idx].reshape(28, 28)
        true_label = y_true[idx]

        x_single = X[idx].reshape(1, -1)
        pred_label = model.predict(x_single)[0]

        plt.subplot(3, 3, i)
        plt.imshow(img, cmap="gray")
        plt.title(f"Prawda: {true_label}\nPred: {pred_label}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def predict_custom_image(model: MLPClassifier, image_path: str):
    canvas = preprocess_image_to_canvas28(image_path)
    x_flat = canvas.reshape(1, -1)

    pred_label = model.predict(x_flat)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_flat)[0]
    else:
        proba = None

    return pred_label, proba, canvas

def main():
    X_train, y_train = load_custom_data("trainingset", LETTER_LABEL_MAP)
    if X_train.shape[0] == 0:
        print("No data available for training. Please ensure the 'trainingset/' directory exists and contains the appropriate subdirectories.")
        return

    # Definicja sieci
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        batch_size=128,
        learning_rate_init=0.001,
        max_iter=20,
        random_state=42,
        verbose=True,
    )

    print("\n== Trening MLP na wybranym zbiorze treningowym ==")
    mlp.fit(X_train, y_train)
    print("KLASY MODELU:", mlp.classes_)

    while True:
        image_path = input("Podaj pełną ścieżkę do pliku z obrazem:\n> ").strip()
        if not os.path.isfile(image_path):
            print("Plik nie istnieje.")
            continue

        try:
            pred_label, proba, arr = predict_custom_image(mlp, image_path)
            print(f"\nSieć przewiduje literę: {pred_label}")
            if proba is not None:
                print("Prawdopodobieństwa klas:")
                for cls, p in zip(mlp.classes_, proba):
                    print(f"  {cls}: {p:.3f}")

            plt.figure()
            plt.imshow(arr, cmap="gray")
            plt.title(f"Przewidywana litera: {pred_label}")
            plt.axis("off")
            plt.show()

        except Exception as e:
            print(f"Wystąpił błąd: {e}")

if __name__ == "__main__":
    main()