# CSV2TFREC
generate a TF record file and label map file from csv files

This application can be fired from the command line to generate Tensorflow record files and label maps for object detection.  

The csv file containing details of images must have the folowing fields  
xmin, ymin, xmax, ymax: These are the bounding box details.  
class: The class ID. The ID's must correspond to the ID's in the labels csv file.  
Path: Path of the image.  
<br>
The csv file containing labels must have the following fields  
class id: class id of the label. The ID's must be same as the ID's in the class field of the csv file containing details of images.  
class label: label of the class.  
<br>
examples:  
This example generates only Train.record and label_map.pbtxt  
CSV2TFREC.py -train_c Train.csv -l labels.csv  
<br>
This example generates Train.record, Test.record and label_map.pbtxt  
CSV2TFREC.py -train_c Train.csv -test_c Test.csv -l labels.csv  
