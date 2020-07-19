from os import walk

import pandas as pd
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from sklearn.decomposition import IncrementalPCA

IMG_HEIGHT = 64
IMG_WIDTH = 64
BATCH_SIZE = 32
IMG_SHAPE = [64, 64, 3]
contamination = 0.4
epochs = 30

inputs_dir = [("./extracted_feature/flickr_DenseNet121.csv", "./datasets/flickr/labels.csv"),
              ("./extracted_feature/flickr_DenseNet169.csv", "./datasets/flickr/labels.csv"),
              ("./extracted_feature/flickr_DenseNet201.csv", "./datasets/flickr/labels.csv"),
              ("./extracted_feature/flickr_ResNet50V2.csv", "./datasets/flickr/labels.csv"),
              ("./extracted_feature/flickr_ResNet101V2.csv", "./datasets/flickr/labels.csv"),
              ("./extracted_feature/flickr_ResNet152V2.csv", "./datasets/flickr/labels.csv")]


def get_list_of_files(temp_data_dir):
    f = []
    for (dirpath, dirnames, filenames) in walk(temp_data_dir):
        f.extend(filenames)
        break
    f.sort()
    return f


def print_score(pyod_labels, labels):
    files = get_list_of_files('/home/alireza/Desktop/images/temp/')
    count = 0
    correct_human = 0
    correct_not_human = 0
    wrong_human = 0
    wrong_not_human = 0
    correct = 0
    for file in files:
        is_human = labels.loc[file][0] == 1
        is_not_human = labels.loc[file][0] == 0
        output = pyod_labels[count]
        if is_human:
            if output == 0:
                correct_human += 1
            else:
                wrong_human += 1
        if is_not_human:
            if output == 1:
                correct_not_human += 1
            else:
                wrong_not_human += 1
        count += 1

    correct = correct_human + correct_not_human

    print("count", count)
    print("correct", correct)
    print("correct_human", correct_human)
    print("correct_not_human", correct_not_human)
    print("wrong_human", wrong_human)
    print("wrong_not_human", wrong_not_human)
    print("accuracy", correct / count)


def run_all_models(all_array, labels, pca):
    if pca:
        transformer = IncrementalPCA(batch_size=100)
        all_array = transformer.fit_transform(all_array)

    clf = OCSVM(contamination=contamination)
    clf.fit(all_array)
    print("OCSVM")
    print_score(clf.labels_, labels)

    clf = AutoEncoder(epochs=epochs, contamination=contamination)
    clf.fit(all_array)
    print("Auto-encoder")
    print_score(clf.labels_, labels)

    clf = HBOS()
    clf.fit(all_array)
    print("HBOS")
    print_score(clf.labels_, labels)

    clf = IForest()
    clf.fit(all_array)
    print("IForest")
    print_score(clf.labels_, labels)

    clf = KNN()
    clf.fit(all_array)
    print("KNN")
    print_score(clf.labels_, labels)

    clf = PCA()
    clf.fit(all_array)
    print("PCA")
    print_score(clf.labels_, labels)


for input_data in inputs_dir:
    labels = pd.read_csv(input_data[1], index_col="img")
    labels = labels.sort_index()
    all_array = pd.read_csv(input_data[0])
    # run_all_models(all_array, labels, False)
    # run_all_models(all_array, labels, True)
