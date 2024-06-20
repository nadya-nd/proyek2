import os
from pyexpat import model
from xml.sax.handler import all_features
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from skimage import filters
from skimage.feature import local_binary_pattern
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def median_filtering(image, kernel_size):
    return cv2.medianBlur(image, kernel_size)

def gaussian_filtering(image, kernel_size, sigma):
    return cv2.GaussianBlur(image, kernel_size, sigma)

# def color_space_transformation(image):
#     transformed_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab) 
    # return transformed_image

def resize_image(image, target_size):
    return cv2.resize(image, target_size)


def histogram_equalization(image):
    # Konversi gambar ke grayscale jika belum grayscale
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()

    # Melakukan histogram equalization
    equalized_image1 = cv2.equalizeHist(gray_image)
    equalized_image2 = cv2.equalizeHist(gray_image)
    return equalized_image1, equalized_image2   


def preprocess_image(image_path, target_size=(64, 64)):
    # Baca gambar
    image = cv2.imread(image_path)

    # Konversi gambar ke grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Penghilangan noise menggunakan Median Filtering
    median_filtered_image = median_filtering(gray_image, 5)

    # # Penghilangan noise menggunakan Gaussian Filtering
    # gaussian_filtered_image = gaussian_filtering(gray_image, (5, 5), 0)

    # # Peningkatan citra menggunakan Histogram Equalization
    # # equalized_image1 = histogram_equalization(median_filtered_image)
    # # equalized_image2 = histogram_equalization(gaussian_filtered_image)
    # equalized_image1, equalized_image2 = histogram_equalization(median_filtered_image, gaussian_filtered_image)

    # Penghilangan noise menggunakan Gaussian Filtering
    gaussian_filtered_image = gaussian_filtering(median_filtered_image, (5, 5), 0)

    
    # Mengubah ukuran citra
    resized_image = resize_image(gaussian_filtered_image,target_size)

    # Peningkatan citra menggunakan Histogram Equalization
    equalized_image = histogram_equalization(resized_image)

    return median_filtered_image, gaussian_filtered_image, equalized_image

    # return median_filtered_image, gaussian_filtered_image, equalized_image1, equalized_image2

def segment_image(image):
    # Mengubah citra ke dalam ruang warna Lab (L*a*b*)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reshape citra ke dalam format yang dapat digunakan oleh algoritma K-Means
    reshaped_image = gray_image.reshape((-1, 1))

    # Melakukan clustering menggunakan algoritma K-Means
    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(reshaped_image)

    # Mendapatkan label hasil clustering
    cluster_labels = kmeans.labels_

    # Mendapatkan centroid untuk setiap kelompok
    cluster_centers = kmeans.cluster_centers_

    # Merestruktur citra berdasarkan label kluster
    # segmented_image = cluster_centers[cluster_labels].reshape(image.shape)
    segmented_image = np.uint8(cluster_centers[cluster_labels].reshape(gray_image.shape))

    # # Konversi citra hasil segmentasi ke dalam ruang warna abu-abu
    # gray_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

    # Deteksi tepi menggunakan metode Canny
    edge_detected_image = cv2.Canny(segmented_image, 30, 150)

    return edge_detected_image

def extract_color_features(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    means = cv2.meanStdDev(image_rgb)
    color_features = np.concatenate([means[0], means[1]]).flatten()
    return color_features

def extract_texture_features(image):
    gray_image = img_as_ubyte(rgb2gray(image))
    
    # Ekstraksi fitur tekstur menggunakan Local Binary Patterns (LBP)
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-8)
    
    return hist.tolist()

# def create_cnn_model():
#     # Membuat model CNN
#     model = Sequential()

#     # Layer konvolusi pertama
#     model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))

#     # Layer konvolusi kedua
#     model.add(Conv2D(64, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))

#     # Flattening
#     model.add(Flatten())

#     # Fully connected layer
#     model.add(Dense(units=128, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(units=1, activation='sigmoid'))

#     # Compile model
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#     # Melihat ringkasan model
#     model.summary()
    
#     return model

def create_cnn_model(input_shape):
    # Membuat model CNN
    model = Sequential()

    # Layer konvolusi pertama
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer konvolusi kedua
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flattening
    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))

    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Melihat ringkasan model
    model.summary()
    
    return model

def load_data(root_folder, target_size=(64, 64)):
    X = []
    y = []
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            label = 1 if 'ripe' in folder_name else 0  # Example labeling logic
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                if os.path.isfile(image_path):
                    _, _, preprocessed_image = preprocess_image(image_path, target_size)
                    X.append(preprocessed_image[0])  # Use one of the equalized images
                    y.append(label)
    X = np.array(X)
    y = np.array(y)
    X = X.reshape(-1, target_size[0], target_size[1], 1)  # Add channel dimension for grayscale images
    return X, y

def main():
    # Path ke folder yang berisi folder-folder dengan gambar
    root_folder = 'D:\\imgprcssng\\tomato-new'
    output_folder = 'D:\\imgprcssng\\new'
    
    # Loop melalui setiap folder di root folder
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        
        # Pastikan path adalah sebuah folder
        if os.path.isdir(folder_path):
            print(f'Processing images in folder: {folder_name}')

            # Loop melalui setiap gambar di dalam folder
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                
                # Pastikan path adalah sebuah file gambar
                if os.path.isfile(image_path):
                    image=cv2.imread(image_path)
                    # Praproses gambar
                    # median_filtered_image, gaussian_filtered_image = preprocess_image(image_path)
                    median_filtered_image, gaussian_filtered_image, equalized_image = preprocess_image(image_path)

                    # Simpan hasil praproses ke file di dalam folder output
                    median_output_path = os.path.join(output_folder, 'median_' + image_name)
                    gaussian_output_path = os.path.join(output_folder, 'gaussian_' + image_name)

                    cv2.imwrite(median_output_path, median_filtered_image)
                    cv2.imwrite(gaussian_output_path, gaussian_filtered_image)
                    
                    # print('Preprocessing selesai.')
                    # print("=========================================================")

                    # Segmentasi citra
                    segmented_image = segment_image(image)
                    segmented_output_path = os.path.join(output_folder, 'segmented_' + image_name)
                    cv2.imwrite(segmented_output_path, segmented_image)

                    # print('Segmentasi selesai.')
                    # print("=========================================================")

                     # Ekstraksi fitur warna
                    color_features = extract_color_features(image)
                    
                    # Ekstraksi fitur tekstur
                    texture_features = extract_texture_features(image)

                    print("Color Features:", color_features)
                    
                    print("Texture Features:", texture_features)

                    # menyimpan fitur ke dalam file CSV
                    output_file_path = os.path.join(output_folder, image_name.replace('.jpg', '_features.csv'))
                    with open(output_file_path, 'w') as f:
                        f.write('Color Features\n')
                        f.write(','.join(map(str, color_features)) + '\n')
                        f.write('Texture Features\n')
                        f.write(','.join(map(str, texture_features)) + '\n')
                    



    #                 # Membuat model CNN
    #                 X, y = load_data(root_folder)
    #                 cnn_model = create_cnn_model()
    #                 # model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

    #                 # # Evaluasi model menggunakan data tes
    #                 # loss, accuracy = model.evaluate(X_test, y_test)

    #                 # # Mencetak akurasi di layar
    #                 # print("Test Accuracy:", accuracy)

    #                 # cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

    #                 # loss, accuracy = cnn_model.evaluate(X_test, y_test)
    #                 # print(f"Test Accuracy: {accuracy}")
    #                 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #                 cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    #                 loss, accuracy = cnn_model.evaluate(X_test, y_test)

    # print(f"Test Accuracy: {accuracy}")

                        X, y = load_data(root_folder)
                        if X.size == 0 or y.size == 0:
                            print("No data loaded. Please check the output folder and the preprocessing steps.")
                            return

                        cnn_model = create_cnn_model(input_shape=(64, 64, 1))
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
                        loss, accuracy = cnn_model.evaluate(X_test, y_test)
                        print(f"Test Accuracy: {accuracy}")

if __name__ == "__main__":
    main()

