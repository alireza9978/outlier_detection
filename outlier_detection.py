import pandas as pd

from pyod.models.auto_encoder import AutoEncoder
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from sklearn.decomposition import IncrementalPCA

inputs_dir = [("./extracted_feature/flickr_DenseNet121.csv", "./datasets/flickr/labels.csv"),
              ("./extracted_feature/flickr_DenseNet169.csv", "./datasets/flickr/labels.csv"),
              ("./extracted_feature/flickr_DenseNet201.csv", "./datasets/flickr/labels.csv"),
              ("./extracted_feature/flickr_ResNet50V2.csv", "./datasets/flickr/labels.csv"),
              ("./extracted_feature/flickr_ResNet101V2.csv", "./datasets/flickr/labels.csv"),
              ("./extracted_feature/flickr_ResNet152V2.csv", "./datasets/flickr/labels.csv")]
output_table = []


def print_score(picture_names, pyod_labels, labels):
    count = 0
    correct_human = 0
    correct_not_human = 0
    wrong_human = 0
    wrong_not_human = 0
    for file in picture_names:
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
    print("correct_inlier", correct_human)
    print("correct_outlier", correct_not_human)
    print("wrong_inlier", wrong_human)
    print("wrong_outlier", wrong_not_human)
    print("accuracy", correct / count)
    return correct / count


def run_all_models(all_array, labels, pca, dataset_name):
    picture_name = all_array.get("# img", 1)
    all_array = all_array.drop("# img", 1)

    if pca:
        transformer = IncrementalPCA()
        all_array = transformer.fit_transform(all_array)

    clf = OCSVM()
    clf.fit(all_array)
    print("OCSVM")
    temp = print_score(picture_name, clf.labels_, labels)
    output_table.append(("OCSVM", all_array.shape, temp, dataset_name))

    # clf = AutoEncoder(epochs=30)
    # clf.fit(all_array)
    # print("Auto-encoder")
    # temp = print_score(picture_name, clf.labels_, labels)
    # output_table.append(("Auto-encoder", all_array.shape, temp, dataset_name))

    clf = HBOS()
    clf.fit(all_array)
    print("HBOS")
    temp = print_score(picture_name, clf.labels_, labels)
    output_table.append(("HBOS", all_array.shape, temp, dataset_name))

    clf = IForest()
    clf.fit(all_array)
    print("IForest")
    temp = print_score(picture_name, clf.labels_, labels)
    output_table.append(("IFrorest", all_array.shape, temp, dataset_name))

    clf = KNN()
    clf.fit(all_array)
    print("KNN")
    temp = print_score(picture_name, clf.labels_, labels)
    output_table.append(("KNN", all_array.shape, temp, dataset_name))

    clf = PCA()
    clf.fit(all_array)
    print("PCA")
    temp = print_score(picture_name, clf.labels_, labels)
    output_table.append(("PCA", all_array.shape, temp, dataset_name))


for input_data in inputs_dir:
    labels = pd.read_csv(input_data[1], index_col="img")
    labels = labels.sort_index()
    all_array = pd.read_csv(input_data[0])
    name = str(input_data[0]).split("/")[-1]
    run_all_models(all_array, labels, False, name)
    run_all_models(all_array, labels, True, name)

print(output_table)
