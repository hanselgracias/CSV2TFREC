import argparse
import os
import io
import pandas as pd
import ast
import csv
import tensorflow.compat.v1 as tf
from collections import namedtuple
from PIL import Image
from object_detection.utils import dataset_util, label_map_util
from pathlib import Path

parser = argparse.ArgumentParser(description = "CSV to TF Record Convertor")

parser.add_argument("-train_c", "--train_csv", help="Path of train csv file to be converted to TF record",
                    type=str)
parser.add_argument("-test_c", "--test_csv", help="Path of test csv file to be converted to TF record",
                    type=str, default=None)
parser.add_argument("-l", "--labels", help="Path to labels file",
                    type=str)
parser.add_argument("-o", "--output_dir", help="Path to folder for TF record",
                    type=str, default=None)

args = parser.parse_args()

if(args.output_dir is None):
    args.output_dir = os.path.join(os.getcwd())
    
#convert csv file to labels dictionary 
def csv2labels_dict(file):
    
    keys = []
    values = []
    row_num = 0
    labels_dict = {}
    labels = []

    offset = 0
    
    with open(args.labels) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        for row in csv_reader:
            if (row_num == 1):
                if (int(row[0]) == 0):
                    offset = 1
                    break
            row_num += 1
                
        row_num = 0

    with open(args.labels) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        for row in csv_reader:
            if (row_num != 0):
                keys.append(int(row[0]) + offset)
                values.append(row[1])

            row_num += 1

            
        with open(os.path.join(args.output_dir, "label_map.pbtxt"), 'w') as f:
            for i in keys:
                labels_dict[i] = values[int(i)- offset]
                f.write('item { \n')
                f.write('\tname:\'{}\'\n'.format(values[int(i) - offset]))
                f.write('\tid:{}\n'.format(int(i)))
                f.write('}\n')

        return labels_dict, offset



#convert csv file to pd dataframe
def csv2GroupedData(file, offset):
    df = pd.read_csv(file)
    df["class"] = df["class"] + offset

    grouped_data = namedtuple("grouped_data", ["path", "Obj_PicParam"])
    grouped = df.groupby("Path")

    return([grouped_data(path, grouped.get_group(x)) for path, x in zip(grouped.groups.keys(), grouped.groups.keys())])



#generate TF Record
def create_tf_record(group, path, labels, csv_file):
    with tf.gfile.GFile(os.path.join(os.path.dirname(csv_file),path), 'rb') as fid:
        encoded_img = fid.read()
    encoded_img_io = io.BytesIO(encoded_img)
    image = Image.open(encoded_img_io)
    width, height = image.size
    
    filename = os.path.split(path)[1]
    file_extension = str(os.path.splitext(path)[1]).replace(".", "")

    filename = filename.encode('utf8')
    image_format = bytes(file_extension, encoding="utf8")

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.Obj_PicParam.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(labels[int(row["class"])].encode('utf8'))
        classes.append(int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_img),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example
    
    

def main(_):

    labels, offset = csv2labels_dict(args.labels)
    
    writer = tf.python_io.TFRecordWriter(os.path.join(args.output_dir, "{}.record".format(Path(args.train_csv).stem)))
    grouped = csv2GroupedData(args.train_csv, offset)
    for group in grouped:
        tf_example = create_tf_record(group, group.path, labels, args.train_csv)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print('Successfully created the TFRecord file: {}'.format(os.path.join(args.output_dir, "{}.record".format(Path(args.train_csv).stem))))

    if(args.test_csv is not None):
        writer = tf.python_io.TFRecordWriter(os.path.join(args.output_dir, "{}.record".format(Path(args.test_csv).stem)))
        grouped = csv2GroupedData(args.test_csv, offset)
        for group in grouped:
            tf_example = create_tf_record(group, group.path, labels, args.test_csv)
            writer.write(tf_example.SerializeToString())
        writer.close()
        print('Successfully created the TFRecord file: {}'.format(os.path.join(args.output_dir, "{}.record".format(Path(args.test_csv).stem))))


if __name__ == '__main__':
    tf.app.run()    
    



    





