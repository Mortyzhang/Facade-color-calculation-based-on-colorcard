import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
import csv
import pandas as pd


def get_name(root):
    for root, dirs, file in os.walk(root):
        return file


def read_img(img_root):
    img = cv2.imread(img_root)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    color = [1, 1, 1]
    mask = np.logical_and(*list([img[:, :, i] != color[i] for i in range(3)]))
    return img[mask].reshape((-1, 3))


def make_cluster(data):
    kmeans = KMeans(n_clusters=cluster_number, random_state=0).fit(data)
    center = kmeans.cluster_centers_
    prediction = kmeans.predict(data)
    return center, prediction


def write_record(records):
    name_list = ["img_name"] + ["dominant_color"] +["secondary_color"] + ["ratio_" + str(i+1) for i in range(2)]
    f_val = open("analysis.csv", "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(f_val)
    csv_writer.writerow(name_list)
    for i in range(len(records)):
        write_things = [records[i][0]] + [x for x in list(records[i][1])[:2]] + [y for y in list(records[i][2])[:2]]
        csv_writer.writerow(write_things)
    f_val.close()


def cal_euclidean(center, color):
    new_center = []
    for c in center:
        c = np.array(c)
        c = c[None, :]
        dis = ((c - color) ** 2).sum(-1)
        ids = np.argsort(-dis, axis=0)
        new_center.append(color[ids[0]])
    return new_center


def main():
    color_code = pd.read_csv("ChinaBuildingColorCard_1026_.csv", dtype=str)
    color_data = color_code[["Code", "RGB"]].values

    code_record = []
    color_record = []
    for s in range(len(color_data)):
        if color_data[s][0] == "0":
            continue
        ccc = color_data[s][1].split(" ")
        new_ccc = [int(ccc[0]), int(ccc[1]), int(ccc[2])]
        color_record.append(new_ccc)
        code_record.append(color_data[s][0])

    root_list = get_name(img_folder)
    record = []
    for rr in root_list:
        ratio = [0 for i in range(cluster_number)]
        current_root = img_folder + rr
        img_pixels = read_img(current_root)
        centers, prediction = make_cluster(img_pixels)
        for pp in prediction:
            ratio[int(pp)] += 1
        ratio = [round(ratio[i]/len(prediction), 4) for i in range(len(ratio))]
        centers = np.array(centers, dtype=np.int32)
        translated_centers = cal_euclidean(centers, np.array(color_record))
        record.append([rr, translated_centers, ratio])
    write_record(record)


if __name__ == '__main__':
    img_folder = "images/"
    cluster_number = 8
    img_size = 300
    main()