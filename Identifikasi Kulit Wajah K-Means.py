import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Fungsi untuk memuat gambar sesuai dengan ekstensi file
def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File tidak ditemukan: {image_path}")

    valid_extensions = ['.jpg', '.jpeg', '.png', 'webp']
    ext = os.path.splitext(image_path)[-1].lower()

    if ext not in valid_extensions:
        raise ValueError("Format file tidak didukung. Harap gunakan file dengan ekstensi .jpg, .jpeg, .png, atau .webp.")

    image = Image.open(image_path)
    return image

# Fungsi untuk mengonversi gambar ke grayscale
def convert_to_grayscale(image_np):
    if image_np.shape[2] != 3:
        raise ValueError("Gambar bukan format RGB")
    gray_image = 0.2989 * image_np[:, :, 0] + 0.5870 * image_np[:, :, 1] + 0.1140 * image_np[:, :, 2]
    return gray_image

# Fungsi untuk menampilkan gambar
def display_images(original, grayscale, clustered, result, result_title):
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 4, 1)
    plt.imshow(original)
    plt.axis('off')
    plt.title('Citra Asli')

    plt.subplot(1, 4, 2)
    plt.imshow(grayscale, cmap='gray')
    plt.axis('off')
    plt.title('Citra Grayscale')

    plt.subplot(1, 4, 3)
    plt.imshow(clustered, cmap='gray')
    plt.axis('off')
    plt.title('Hasil K-Means Clustering')

    plt.subplot(1, 4, 4)
    plt.imshow(result)
    plt.axis('off')
    plt.title(result_title)

    plt.show()

# Kelas untuk K-Means
class KMeans:
    def __init__(self, k, seed=None):
        self.k = k
        self.seed = seed

    def train(self, X, MAXITER=100, TOL=1e-3):
        if self.seed is not None:
            np.random.seed(self.seed)
        centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        centroids_old = centroids.copy()
        for iter_ in range(MAXITER):
            dist = np.linalg.norm(X - centroids[0, :], axis=1).reshape(-1, 1)
            for class_ in range(1, self.k):
                dist = np.append(dist, np.linalg.norm(X - centroids[class_, :], axis=1).reshape(-1, 1), axis=1)
            classes = np.argmin(dist, axis=1)
            for class_ in set(classes):
                centroids[class_, :] = np.mean(X[classes == class_, :], axis=0)
            if np.linalg.norm(centroids - centroids_old) < TOL:
                break
            centroids_old = centroids.copy()
        self.centroids = centroids

    def predict(self, X):
        dist = np.linalg.norm(X - self.centroids[0, :], axis=1).reshape(-1, 1)
        for class_ in range(1, self.k):
            dist = np.append(dist, np.linalg.norm(X - self.centroids[class_, :], axis=1).reshape(-1, 1), axis=1)
        classes = np.argmin(dist, axis=1)
        return classes

# Fungsi untuk melakukan clustering dengan K-Means
def perform_kmeans(gray_image, k, seed):
    pixels = gray_image.reshape(-1, 1)
    kmeans = KMeans(k, seed=seed)
    kmeans.train(pixels)
    classes = kmeans.predict(pixels)
    clustered_image = classes.reshape(gray_image.shape)
    return clustered_image, kmeans.centroids

# Fungsi untuk menentukan hasil akhir berdasarkan clustering
def determine_result(image_np, clustered_image, kmeans_centroids):
    unique, counts = np.unique(clustered_image, return_counts=True)
    cluster_frequencies = dict(zip(unique, counts))
    dominant_cluster = max(cluster_frequencies, key=cluster_frequencies.get)

    if dominant_cluster == 1:
        result_title = 'Wajah Normal'
        result_image = image_np
    else:
        acne_cluster = np.argmin(kmeans_centroids)
        acne_mask = (clustered_image == acne_cluster).astype(np.uint8)
        result_title = 'Wajah Berjerawat'
        result_image = image_np.copy()
        result_image[acne_mask == 1] = [255, 0, 0]
    return result_image, result_title

# Menyesuaikan dengan path image
image_path = 'path_direktori_Anda/wajah.jpg'

# Main execution
image = load_image(image_path)
image_np = np.array(image)
gray_image = convert_to_grayscale(image_np)
clustered_image, kmeans_centroids = perform_kmeans(gray_image, k=3, seed=42)
result_image, result_title = determine_result(image_np, clustered_image, kmeans_centroids)

display_images(image, gray_image, clustered_image, result_image, result_title)