import numpy as np
import cv2
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


original = os.getcwd()
os.chdir('Dataset')
cur = os.getcwd()
os.chdir(original)

train_folder_face = os.path.join(cur, 'train_face_images/')
train_folder_nonface = os.path.join(cur, 'train_nonface_images/')
test_folder_face = os.path.join(cur, 'test_face_images/')
test_folder_nonface = os.path.join(cur, 'test_nonface_images/')

files_train_face = os.listdir(train_folder_face)
files_train_nonface = os.listdir(train_folder_nonface)
files_test_face = os.listdir(test_folder_face)
files_test_nonface = os.listdir(test_folder_nonface)




"Returns single image at given index"

def get_image_train(index, typ, gray=0, img_size=60):
    dat = []
    if typ == 'face':
        dat = cv2.imread(train_folder_face + files_train_face[index])
    elif typ == 'nonface':
        dat = cv2.imread(train_folder_nonface + files_train_nonface[index])
    if gray:
        dat = cv2.cvtColor(dat, cv2.COLOR_BGR2GRAY).astype('float')
    dat = cv2.resize(dat, (img_size, img_size))
    return dat



def get_image_train2(index, typ, gray=0, img_size=60):
    dat = temp = []
    if typ == 'face':
        dat = cv2.imread(train_folder_face + files_train_face[index])
    elif typ == 'nonface':
        dat = cv2.imread(train_folder_nonface + files_train_nonface[index])
    if gray:
        dat = cv2.cvtColor(dat, cv2.COLOR_BGR2GRAY).astype('float')
    dat = cv2.resize(dat, (img_size, img_size))
    if gray:
        temp = np.zeros((img_size, img_size, 1))
        temp[:, :, 0] = dat
    else:
        temp = dat
    return temp


"Returns single image at given index"

def get_image_CV(index, typ, gray=0, img_size=60):
    dat = []
    if typ == 'face':
        dat = cv2.imread(test_folder_face + files_test_face[index])
    elif typ == 'nonface':
        dat = cv2.imread(test_folder_nonface + files_test_nonface[index])
    if gray:
        dat = cv2.cvtColor(dat, cv2.COLOR_BGR2GRAY).astype('float')
    dat = cv2.resize(dat, (img_size, img_size))
    return dat


def get_image_CV2(index, typ, gray=0, img_size=60):
    dat = temp = []
    if typ == 'face':
        dat = cv2.imread(test_folder_face + files_test_face[index])
    elif typ == 'nonface':
        dat = cv2.imread(test_folder_nonface + files_test_nonface[index])
    if gray:
        dat = cv2.cvtColor(dat, cv2.COLOR_BGR2GRAY).astype('float')
    dat = cv2.resize(dat, (img_size, img_size))
    if gray:
        temp = np.zeros((img_size, img_size, 1))
        temp[:, :, 0] = dat
    else:
        temp = dat
    return temp


"Displays an image"

def display(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"Returns PCA computed array"

def perform_pca(X, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(X)
    X_PCA = pca.transform(X)
    return X_PCA, pca


"Normalizes the data (Mean=0, Covariance=1)"

def preprocess(X):
    scaler = StandardScaler()
    scaler.fit(X)
    X_PP = scaler.transform(X)
    return X_PP


"Returns an array with all the images concatenated" \
"<Rows are the image>"


def load_train_data(typ, gray=0, len_train=10000):
    X_train = []
    for i in range(0, len_train):
        if typ == 'face':
            dat = get_image_train(i, 'face', gray)
            dat = dat.flatten()
            X_train.append(dat)
        elif typ == 'nonface':
            dat = get_image_train(i, 'nonface', gray)
            dat = dat.flatten()
            X_train.append(dat)
    X_train = np.array(X_train)
    return X_train
    'Each ROW is an IMAGE'


"Returns an array with all the images concatenated" \
"<Rows are the image>"


def load_test_data(typ, gray=0, len_test=1000):
    X_CV = []
    for i in range(0, len_test):
        if typ == 'face':
            dat = get_image_CV(i, 'face', gray)
            dat = dat.flatten()
            X_CV.append(dat)
        elif typ == 'nonface':
            dat = get_image_CV(i, 'nonface', gray)
            dat = dat.flatten()
            X_CV.append(dat)
    X_CV = np.array(X_CV)
    return X_CV






