cd ../../
mkdir images/out/rouen/0
mkdir images/out/rouen/40
mkdir images/out/rouen/70
python test_yolo.py  -c /users/ahjiang/image-data/bb/udacity-od-crowdai/Udacity_object_dataset/crowdai-labels.txt -t /users/ahjiang/image-data/bb/rouen/rouen_frames -o images/out/rouen/0 --model_path /users/ahjiang/src/YAD2K/data/models/exp/yolov2-model-0.6-0fr.h5
#python test_yolo.py  -c /users/ahjiang/image-data/bb/udacity-od-crowdai/Udacity_object_dataset/crowdai-labels.txt -t /users/ahjiang/image-data/bb/rouen/rouen_frames -o images/out/rouen/10 --model_path /users/ahjiang/src/YAD2K/data/models/exp/yolov2-model-0.6-10fr.h5
#python test_yolo.py  -c /users/ahjiang/image-data/bb/udacity-od-crowdai/Udacity_object_dataset/crowdai-labels.txt -t /users/ahjiang/image-data/bb/rouen/rouen_frames -o images/out/rouen/20 --model_path /users/ahjiang/src/YAD2K/data/models/exp/yolov2-model-0.6-20fr.h5
#python test_yolo.py  -c /users/ahjiang/image-data/bb/udacity-od-crowdai/Udacity_object_dataset/crowdai-labels.txt -t /users/ahjiang/image-data/bb/rouen/rouen_frames -o images/out/rouen/30 --model_path /users/ahjiang/src/YAD2K/data/models/exp/yolov2-model-0.6-30fr.h5
python test_yolo.py  -c /users/ahjiang/image-data/bb/udacity-od-crowdai/Udacity_object_dataset/crowdai-labels.txt -t /users/ahjiang/image-data/bb/rouen/rouen_frames -o images/out/rouen/40 --model_path /users/ahjiang/src/YAD2K/data/models/exp/yolov2-model-0.6-40fr.h5
#python test_yolo.py  -c /users/ahjiang/image-data/bb/udacity-od-crowdai/Udacity_object_dataset/crowdai-labels.txt -t /users/ahjiang/image-data/bb/rouen/rouen_frames -o images/out/rouen/50 --model_path /users/ahjiang/src/YAD2K/data/models/exp/yolov2-model-0.6-50fr.h5
#python test_yolo.py  -c /users/ahjiang/image-data/bb/udacity-od-crowdai/Udacity_object_dataset/crowdai-labels.txt -t /users/ahjiang/image-data/bb/rouen/rouen_frames -o images/out/rouen/60 --model_path /users/ahjiang/src/YAD2K/data/models/exp/yolov2-model-0.6-60fr.h5
python test_yolo.py  -c /users/ahjiang/image-data/bb/udacity-od-crowdai/Udacity_object_dataset/crowdai-labels.txt -t /users/ahjiang/image-data/bb/rouen/rouen_frames -o images/out/rouen/70 --model_path /users/ahjiang/src/YAD2K/data/models/exp/yolov2-model-0.6-70fr.h5
