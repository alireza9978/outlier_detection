from time import time

import pandas as pd
import xlsxwriter
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.mcd import MCD
from pyod.models.mo_gaal import MO_GAAL
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.so_gaal import SO_GAAL
from pyod.models.sos import SOS
from pyod.utils.utility import precision_n_scores
from pyod.utils.utility import standardizer
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

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
    worksheet.write(0, 5, "execution_time", bold)
    worksheet.write(0, 6, "score_ROC", bold)
    worksheet.write(0, 7, "score_PRN", bold)
    worksheet.write(0, 8, "score_acc", bold)

    for item in output_table:
        _ = str(item[3]).split(".")[0]
        feature_extractor_model = _.split("_")[1]
        data_set_name = _.split("_")[0]
        worksheet.write(row, 0, data_set_name)
        worksheet.write(row, 1, feature_extractor_model)
        worksheet.write(row, 2, item[1][1])
        worksheet.write(row, 3, item[1][0])
        worksheet.write(row, 4, item[0])
        worksheet.write(row, 5, item[4])
        worksheet.write(row, 6, item[2][0])
        worksheet.write(row, 7, item[2][1])
        worksheet.write(row, 8, item[2][2])
        row += 1

    workbook.close()


def print_score(picture_names, test_scores, y_test):
    count = 0
    correct_human = 0
    correct_not_human = 0
    wrong_human = 0
    wrong_not_human = 0
    for file in picture_names:
        is_human = data_set_labels.loc[file][0] == 1
        is_not_human = data_set_labels.loc[file][0] == 0

        output = test_scores[count]
        if is_human:
            if output > 0.50:
                correct_human += 1
            else:
                wrong_human += 1
        if is_not_human:
            if output > 0.50:
                wrong_not_human += 1
            else:
                correct_not_human += 1
        count += 1

    correct = correct_human + correct_not_human
    roc = round(roc_auc_score(y_test, test_scores), ndigits=4)
    prn = round(precision_n_scores(y_test, test_scores), ndigits=4)
    return roc, prn, correct / count


def run_all_models(all_array, labels, pca, data_set_name):
    picture_name = all_array.get("# img", 1)
    all_array = all_array.drop("# img", 1)

    # standardizing data for processing
    all_array = standardizer(all_array)

    y = labels.get("in").to_numpy()
    x_train, x_test, y_train, y_test, picture_train, picture_test = train_test_split(all_array, y, picture_name,
                                                                                     test_size=0.4)

    if pca:
        transformer = IncrementalPCA()
        all_array = transformer.fit_transform(all_array)

    print("OCSVM")
    now = time()
    clf = OCSVM()
    clf.fit(x_train)
    test_scores = clf.decision_function(x_test)
    temp = print_score(picture_test, test_scores, y_test)
    output_table.append(("OCSVM", all_array.shape, temp, data_set_name, time() - now))

    # print("Auto-encoder")
    # clf = AutoEncoder(epochs=30)
    # clf.fit(x_train)
    # test_scores = clf.decision_function(x_test)
    # temp = print_score(picture_test, test_scores, y_test)
    # output_table.append(("Auto-encoder", all_array.shape, temp, dataset_name,time() - now))

    print("HBOS")
    now = time()
    clf = HBOS()
    clf.fit(x_train)
    test_scores = clf.decision_function(x_test)
    temp = print_score(picture_test, test_scores, y_test)
    output_table.append(("HBOS", all_array.shape, temp, data_set_name, time() - now))

    print("SO_GAAL")
    now = time()
    clf = SO_GAAL()
    clf.fit(x_train)
    test_scores = clf.decision_function(x_test)
    temp = print_score(picture_test, test_scores, y_test)
    output_table.append(("SO_GAAL", all_array.shape, temp, data_set_name, time() - now))

    print("MO_GAAL")
    now = time()
    clf = MO_GAAL()
    clf.fit(x_train)
    test_scores = clf.decision_function(x_test)
    temp = print_score(picture_test, test_scores, y_test)
    output_table.append(("MO_GAAL", all_array.shape, temp, data_set_name, time() - now))

    print("MCD")
    now = time()
    clf = MCD()
    clf.fit(x_train)
    test_scores = clf.decision_function(x_test)
    temp = print_score(picture_test, test_scores, y_test)
    output_table.append(("MCD", all_array.shape, temp, data_set_name, time() - now))

    print("SOS")
    now = time()
    clf = SOS()
    clf.fit(x_train)
    test_scores = clf.decision_function(x_test)
    temp = print_score(picture_test, test_scores, y_test)
    output_table.append(("SOS", all_array.shape, temp, data_set_name, time() - now))

    print("IForest")
    now = time()
    clf = IForest()
    clf.fit(x_train)
    test_scores = clf.decision_function(x_test)
    temp = print_score(picture_test, test_scores, y_test)
    output_table.append(("IFrorest", all_array.shape, temp, data_set_name, time() - now))

    print("KNN")
    now = time()
    clf = KNN()
    clf.fit(x_train)
    test_scores = clf.decision_function(x_test)
    temp = print_score(picture_test, test_scores, y_test)
    output_table.append(("KNN", all_array.shape, temp, data_set_name, time() - now))

    print("PCA")
    now = time()
    clf = PCA()
    clf.fit(x_train)
    test_scores = clf.decision_function(x_test)
    temp = print_score(picture_test, test_scores, y_test)
    output_table.append(("PCA", all_array.shape, temp, data_set_name, time() - now))


for input_data in inputs_dir:
    data_set_labels = pd.read_csv(input_data[1], index_col="img")
    data_set_labels = data_set_labels.sort_index()
    data_set = pd.read_csv(input_data[0])
    name = str(input_data[0]).split("/")[-1]
    run_all_models(data_set, data_set_labels, False, name)
    run_all_models(data_set, data_set_labels, True, name)

write_to_xlsx()
