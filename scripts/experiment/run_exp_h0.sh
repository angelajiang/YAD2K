step=5

npz="/datasets/BigLearning/ahjiang/bb/npz/crowdai-pedestrian-training-0.8.npz"
labels="/datasets/BigLearning/ahjiang/bb/udacity-od-crowdai/Udacity_object_dataset/labels/pedestrian-label.txt"
outmodel="data/models/exp/yolov2-model-pedestrian"

for i in `seq 0 $step 70`;
do
    python -u run_train.py  -d $npz -c $labels -p $outmodel -n $i
done
