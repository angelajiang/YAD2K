
npz="/datasets/BigLearning/ahjiang/bb/npz/crowdai-caltech-pedestrian-training.npz"
labels="/datasets/BigLearning/ahjiang/bb/udacity-od-crowdai/Udacity_object_dataset/labels/pedestrian-label.txt"
outmodel="data/models/exp/yolov2-model-pedestrian-augmented"

step=10
start=5
end=74

for i in `seq $start $step $end`;
do
    python -u run_train.py -d $npz -c $labels -p $outmodel -n $i
done
