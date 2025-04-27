import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# === PARAMETERS ===
DATASET_DIR = 'dataset'
MODEL_SAVE_PATH = 'model/eigenface_pipeline.pkl'
FACE_SIZE = (128, 128)

# === Custom Transformer for Mean Centering ===
class MeanCentering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.mean_face = np.mean(X, axis=0)
        return self

    def transform(self, X):
        return X - self.mean_face

# === Face Detection ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image_gray, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
    faces = face_cascade.detectMultiScale(
        image_gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size
    )
    return faces

def crop_faces(image_gray, faces, return_all=False):
    cropped_faces = []
    selected_faces = []
    if len(faces) > 0:
        if return_all:
            for x, y, w, h in faces:
                selected_faces.append((x, y, w, h))
                cropped_faces.append(image_gray[y:y+h, x:x+w])
        else:
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            selected_faces.append((x, y, w, h))
            cropped_faces.append(image_gray[y:y+h, x:x+w])
    return cropped_faces, selected_faces

def resize_and_flatten(face):
    face_resized = cv2.resize(face, FACE_SIZE)
    face_flattened = face_resized.flatten()
    return face_flattened

# === Loading Dataset ===
def load_dataset(dataset_dir):
    X = []
    y = []
    print("[INFO] Loading dataset...")

    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        print(f"[INFO] Processing folder: {person_name}")
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            print(f"    Loading image: {img_name}")
            image = cv2.imread(img_path)
            if image is None:
                print(f"[WARNING] Could not read image {img_path}")
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(gray)

            if faces is None or len(faces) == 0:
                print(f"    [WARNING] No face detected in {img_name}")
                continue

            cropped_faces, _ = crop_faces(gray, faces)
            if len(cropped_faces) > 0:
                face_flattened = resize_and_flatten(cropped_faces[0])
                X.append(face_flattened)
                y.append(person_name)

    print(f"[INFO] Finished loading {len(X)} face samples.")
    return np.array(X), np.array(y)

# === Train Model ===
def train_model(X, y):
    print("[INFO] Training model...")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=177, stratify=y)

    pipeline = Pipeline([
        ('centering', MeanCentering()),
        ('pca', PCA(svd_solver='randomized', whiten=True, random_state=177)),
        ('svc', SVC(kernel='linear', random_state=177))
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    return pipeline

# === Save Model ===
def save_model(pipeline, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"[INFO] Model saved to {save_path}")

# === Main Flow ===
def main():
    X, y = load_dataset(DATASET_DIR)
    pipeline = train_model(X, y)
    save_model(pipeline, MODEL_SAVE_PATH)

    # Optional: visualize Eigenfaces
    n_components = len(pipeline[1].components_)
    eigenfaces = pipeline[1].components_.reshape((n_components, X.shape[1]))

    ncol = 4
    nrow = (n_components + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 2.5*nrow), subplot_kw={'xticks':[], 'yticks':[]})

    for i, ax in enumerate(axes.flat):
        if i < n_components:
            ax.imshow(eigenfaces[i].reshape(FACE_SIZE), cmap='gray')
            ax.set_title(f'Eigenface {i+1}')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()