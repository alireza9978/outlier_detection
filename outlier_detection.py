from time import time

import pandas as pd
import xlsxwriter
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


def write_to_xlsx():
    workbook = xlsxwriter.Workbook('result.xlsx')
    worksheet = workbook.add_worksheet()

    row = 1
    bold = workbook.add_format({'bold': True})

    worksheet.set_column(0, 7, 15)
    worksheet.set_column(1, 1, 30)
    worksheet.set_column(4, 4, 40)

    worksheet.write(0, 0, "dataset", bold)
    worksheet.write(0, 1, "feature_extractor_model", bold)
    worksheet.write(0, 2, "feature_count", bold)
    worksheet.write(0, 3, "item_count", bold)
    worksheet.write(0, 4, "outlier_detection_algorithm", bold)
    worksheet.write(0, 5, "score_method", bold)
    worksheet.write(0, 6, "score", bold)
    worksheet.write(0, 7, "execution_time", bold)
    worksheet.center_horizontally()
    worksheet.center_vertically()

    for item in output_table:
        _ = str(item[3]).split(".")[0]
        feature_extractor_model = _.split("_")[1]
        data_set_name = _.split("_")[0]
        worksheet.write(row, 0, data_set_name)
        worksheet.write(row, 1, feature_extractor_model)
        worksheet.write(row, 2, item[1][1])
        worksheet.write(row, 3, item[1][0])
        worksheet.write(row, 4, item[0])
        worksheet.write(row, 5, "Accuracy")
        worksheet.write(row, 6, item[2])
        worksheet.write(row, 7, item[4])
        row += 1

    workbook.close()


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

    now = time()
    clf = OCSVM()
    clf.fit(all_array)
    print("OCSVM")
    temp = print_score(picture_name, clf.labels_, labels)
    output_table.append(("OCSVM", all_array.shape, temp, dataset_name, time() - now))

    # clf = AutoEncoder(epochs=30)
    # clf.fit(all_array)
    # print("Auto-encoder")
    # temp = print_score(picture_name, clf.labels_, labels)
    # output_table.append(("Auto-encoder", all_array.shape, temp, dataset_name,time() - now))

    now = time()
    clf = HBOS()
    clf.fit(all_array)
    print("HBOS")
    temp = print_score(picture_name, clf.labels_, labels)
    output_table.append(("HBOS", all_array.shape, temp, dataset_name, time() - now))

    now = time()
    clf = IForest()
    clf.fit(all_array)
    print("IForest")
    temp = print_score(picture_name, clf.labels_, labels)
    output_table.append(("IFrorest", all_array.shape, temp, dataset_name, time() - now))

    now = time()
    clf = KNN()
    clf.fit(all_array)
    print("KNN")
    temp = print_score(picture_name, clf.labels_, labels)
    output_table.append(("KNN", all_array.shape, temp, dataset_name, time() - now))

    now = time()
    clf = PCA()
    clf.fit(all_array)
    print("PCA")
    temp = print_score(picture_name, clf.labels_, labels)
    output_table.append(("PCA", all_array.shape, temp, dataset_name, time() - now))


for input_data in inputs_dir:
    labels = pd.read_csv(input_data[1], index_col="img")
    labels = labels.sort_index()
    all_array = pd.read_csv(input_data[0])
    name = str(input_data[0]).split("/")[-1]
    run_all_models(all_array, labels, False, name)
    run_all_models(all_array, labels, True, name)

write_to_xlsx()
