step=5

npz="/datasets/BigLearning/ahjiang/bb/npz/crowdai-truck-training-0.8.npz"
labels="/datasets/BigLearning/ahjiang/bb/udacity-od-crowdai/Udacity_object_dataset/labels/truck-label.txt"
outmodel="data/models/exp/yolov2-model-truck"

for i in `seq 0 $step 70`;
do
    python -u run_train.py  -d $npz -c $labels -p $outmodel -n $i
done
