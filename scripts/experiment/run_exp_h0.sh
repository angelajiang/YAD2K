cd ../../
python train.py  -d ~/image-data/bb/npz/object-detection-crowdai-0.6.npz -c /users/ahjiang/image-data/bb/udacity-od-crowdai/Udacity_object_dataset/crowdai-labels.txt -p data/models/exp/yolov2-model-0.6 -n 0
python train.py  -d ~/image-data/bb/npz/object-detection-crowdai-0.6.npz -c /users/ahjiang/image-data/bb/udacity-od-crowdai/Udacity_object_dataset/crowdai-labels.txt -p data/models/exp/yolov2-model-0.6 -n 10
python train.py  -d ~/image-data/bb/npz/object-detection-crowdai-0.6.npz -c /users/ahjiang/image-data/bb/udacity-od-crowdai/Udacity_object_dataset/crowdai-labels.txt -p data/models/exp/yolov2-model-0.6 -n 20
