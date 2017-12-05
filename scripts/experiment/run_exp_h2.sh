#npz="/datasets/BigLearning/ahjiang/bb/npz/crowdai-0.8.npz"
#labels="/datasets/BigLearning/ahjiang/bb/udacity-od-crowdai/Udacity_object_dataset/labels/crowdai-labels.txt"
#outmodel="data/models/exp/yolov2-model"

npz="/datasets/BigLearning/ahjiang/bb/npz/crowdai-truck-training-0.8.npz"
labels="/datasets/BigLearning/ahjiang/bb/udacity-od-crowdai/Udacity_object_dataset/labels/truck-label.txt"
outmodel="data/models/exp/yolov2-model-truck"

step=5
start=0
n=74

for i in `seq $start $step $n`;
do
    python -u run_train.py  -d $npz -c $labels -p $outmodel -n $i
done
