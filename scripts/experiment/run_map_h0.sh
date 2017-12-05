echo "hybrid-0"
python -u test_yolo.py  -c /datasets/BigLearning/ahjiang/bb/udacity-od-crowdai/Udacity_object_dataset/crowdai-labels.txt -o images/tmp/ --model_path /users/ahjiang/src/YAD2K/data/models/exp/yolov2-model-hybrid-0fr.h5 --mode 1 -t /datasets/BigLearning/ahjiang/bb/npz/crowdai-test.npz
echo "hybrid-30"
python -u test_yolo.py  -c /datasets/BigLearning/ahjiang/bb/udacity-od-crowdai/Udacity_object_dataset/crowdai-labels.txt -o images/tmp/ --model_path /users/ahjiang/src/YAD2K/data/models/exp/yolov2-model-hybrid-30fr.h5 --mode 1 -t /datasets/BigLearning/ahjiang/bb/npz/crowdai-test.npz
echo "hybrid-70"
python -u test_yolo.py  -c /datasets/BigLearning/ahjiang/bb/udacity-od-crowdai/Udacity_object_dataset/crowdai-labels.txt -o images/tmp/ --model_path /users/ahjiang/src/YAD2K/data/models/exp/yolov2-model-hybrid-70fr.h5 --mode 1 -t /datasets/BigLearning/ahjiang/bb/npz/crowdai-test.npz
echo "split-0"
python -u test_yolo.py  -c /datasets/BigLearning/ahjiang/bb/udacity-od-crowdai/Udacity_object_dataset/crowdai-labels.txt -o images/tmp/ --model_path /users/ahjiang/src/YAD2K/data/models/exp/yolov2-model-split-0fr.h5 --mode 1 -t /datasets/BigLearning/ahjiang/bb/npz/crowdai-test.npz
echo "split-10"
python -u test_yolo.py  -c /datasets/BigLearning/ahjiang/bb/udacity-od-crowdai/Udacity_object_dataset/crowdai-labels.txt -o images/tmp/ --model_path /users/ahjiang/src/YAD2K/data/models/exp/yolov2-model-split-10fr.h5 --mode 1 -t /datasets/BigLearning/ahjiang/bb/npz/crowdai-test.npz
echo "split-20"
python -u test_yolo.py  -c /datasets/BigLearning/ahjiang/bb/udacity-od-crowdai/Udacity_object_dataset/crowdai-labels.txt -o images/tmp/ --model_path /users/ahjiang/src/YAD2K/data/models/exp/yolov2-model-split-20fr.h5 --mode 1 -t /datasets/BigLearning/ahjiang/bb/npz/crowdai-test.npz
echo "split-30"
python -u test_yolo.py  -c /datasets/BigLearning/ahjiang/bb/udacity-od-crowdai/Udacity_object_dataset/crowdai-labels.txt -o images/tmp/ --model_path /users/ahjiang/src/YAD2K/data/models/exp/yolov2-model-split-30fr.h5 --mode 1 -t /datasets/BigLearning/ahjiang/bb/npz/crowdai-test.npz
echo "split-40"
python -u test_yolo.py  -c /datasets/BigLearning/ahjiang/bb/udacity-od-crowdai/Udacity_object_dataset/crowdai-labels.txt -o images/tmp/ --model_path /users/ahjiang/src/YAD2K/data/models/exp/yolov2-model-split-40fr.h5 --mode 1 -t /datasets/BigLearning/ahjiang/bb/npz/crowdai-test.npz
echo "split-50"
python -u test_yolo.py  -c /datasets/BigLearning/ahjiang/bb/udacity-od-crowdai/Udacity_object_dataset/crowdai-labels.txt -o images/tmp/ --model_path /users/ahjiang/src/YAD2K/data/models/exp/yolov2-model-split-50fr.h5 --mode 1 -t /datasets/BigLearning/ahjiang/bb/npz/crowdai-test.npz
echo "split-60"
python -u test_yolo.py  -c /datasets/BigLearning/ahjiang/bb/udacity-od-crowdai/Udacity_object_dataset/crowdai-labels.txt -o images/tmp/ --model_path /users/ahjiang/src/YAD2K/data/models/exp/yolov2-model-split-60fr.h5 --mode 1 -t /datasets/BigLearning/ahjiang/bb/npz/crowdai-test.npz
echo "split-70"
python -u test_yolo.py  -c /datasets/BigLearning/ahjiang/bb/udacity-od-crowdai/Udacity_object_dataset/crowdai-labels.txt -o images/tmp/ --model_path /users/ahjiang/src/YAD2K/data/models/exp/yolov2-model-split-70fr.h5 --mode 1 -t /datasets/BigLearning/ahjiang/bb/npz/crowdai-test.npz
