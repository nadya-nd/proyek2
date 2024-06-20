import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

knn_model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

def preprocess_image(image_path, target_size=(64, 64)):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median_filtered_image = median_filtering(gray_image, 5)
    gaussian_filtered_image = gaussian_filtering(median_filtered_image, (5, 5), 0)
    resized_image = resize_image(gaussian_filtered_image, target_size)
    equalized_image = histogram_equalization(resized_image)
    return gray_image, median_filtered_image, equalized_image

def segment_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reshaped_image = gray_image.reshape((-1, 1))
    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(reshaped_image)
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    segmented_image = np.uint8(cluster_centers[cluster_labels].reshape(gray_image.shape))
    edge_detected_image = cv2.Canny(segmented_image, 30, 150)
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

@app.route('/', methods=['GET', 'POST'])
def index():
    global knn_model
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            gray_image, _, equalized_image = preprocess_image(filepath)
            segmented_image = segment_image(cv2.imread(filepath))
            color_features = extract_color_features(cv2.imread(filepath))
            texture_features = extract_texture_features(cv2.imread(filepath))
            all_features = np.concatenate([color_features, texture_features])
            prediction = knn_model.predict([all_features])[0]
            result = 'Sick' if prediction == 1 else 'Healthy'
            image_url = url_for('static', filename='uploads/' + filename)
            return render_template('result.html', result=result, image_url=image_url)
    return render_template('index.html')

if __name__ == '__main__':
    with open('knn_model.pkl', 'rb') as f:
        knn_model = pickle.load(f)
    app.run(debug=True)
