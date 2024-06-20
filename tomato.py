import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

label_dict = {
    'bacterial_spot': 0,
    'early_blight': 1,
    'healthy': 2,
    'late_blight': 3,
    'leaf_mold': 4,
    'powdery_mildew': 5,
    'septoria_leaf_spot': 6,
    'spider_mites_two_spotted_spider_mite': 7,
    'target_spot': 8,
    'tomato_mosaic_virus': 9,
    'tomato_yellow_leaf_curl_virus': 10
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def median_filtering(image, kernel_size):
    return cv2.medianBlur(image, kernel_size)

def gaussian_filtering(image, kernel_size, sigma):
    return cv2.GaussianBlur(image, kernel_size, sigma)

def resize_image(image, target_size):
    return cv2.resize(image, target_size)

def histogram_equalization(image):
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()
    equalized_image = cv2.equalizeHist(gray_image)
    return equalized_image

def preprocess_image(image_path, target_size=(64, 64), output_folder=None):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median_filtered_image = cv2.medianBlur(gray_image, 5)
    gaussian_filtered_image = cv2.GaussianBlur(median_filtered_image, (5, 5), 0)
    resized_image = cv2.resize(gaussian_filtered_image, target_size)
    equalized_image = cv2.equalizeHist(resized_image)

    if output_folder:
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        cv2.imwrite(os.path.join(output_folder, f"{base_filename}_median.png"), median_filtered_image)
        cv2.imwrite(os.path.join(output_folder, f"{base_filename}_gaussian.png"), gaussian_filtered_image)
        cv2.imwrite(os.path.join(output_folder, f"{base_filename}_resized.png"), resized_image)
        cv2.imwrite(os.path.join(output_folder, f"{base_filename}_equalized.png"), equalized_image)
    
        return equalized_image

def segment_image(image, output_folder=None):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reshaped_image = gray_image.reshape((-1, 1))
    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(reshaped_image)
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    segmented_image = np.uint8(cluster_centers[cluster_labels].reshape(gray_image.shape))
    edge_detected_image = cv2.Canny(segmented_image, 30, 150)

    if output_folder:
        base_filename = os.path.splitext(os.path.basename(image))[0]
        cv2.imwrite(os.path.join(output_folder, f"{base_filename}_segmented.png"), segmented_image)
        cv2.imwrite(os.path.join(output_folder, f"{base_filename}_edges.png"), edge_detected_image)
    
    return edge_detected_image

def extract_color_features(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    means = cv2.meanStdDev(image_rgb)
    color_features = np.concatenate([means[0], means[1]]).flatten()
    return color_features

def extract_texture_features(image):
    gray_image = img_as_ubyte(rgb2gray(image))
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-8)
    return hist.tolist()


def load_data(folder):
    features = []
    labels = []
    for subdir in os.listdir(folder):
        subdir_path = os.path.join(folder, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, filename)
                if allowed_file(filename):
                    image = cv2.imread(file_path)
                    equalized_image = preprocess_image(file_path)  # Hanya mengembalikan satu nilai
                    color_features = extract_color_features(image)
                    texture_features = extract_texture_features(image)
                    all_features = np.concatenate([color_features, texture_features])
                    features.append(all_features)
                    labels.append(label_dict[subdir])
    return np.array(features), np.array(labels)

def train_model():
    train_folder = 'D:\\imgprcssng\\knn\\train'
    X_train, y_train = load_data(train_folder)
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    with open('knn_model.pkl', 'wb') as f:
        pickle.dump(knn_model, f)

def evaluate_model():
    test_folder = 'D:\\imgprcssng\\knn\\testing'
    X_test, y_test = load_data(test_folder)
    with open('knn_model.pkl', 'rb') as f:
        knn_model = pickle.load(f)
    y_pred = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)

     # Output hasil prediksi
    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            print("Daun Sakit")
        else:
            print("Daun Tidak Sakit")

if __name__ == "__main__":
    train_model()
    evaluate_model()
 