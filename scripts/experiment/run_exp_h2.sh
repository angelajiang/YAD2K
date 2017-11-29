#npz="/datasets/BigLearning/ahjiang/bb/npz/crowdai-0.8.npz"
#labels="/datasets/BigLearning/ahjiang/bb/udacity-od-crowdai/Udacity_object_dataset/labels/crowdai-labels.txt"
#outmodel="data/models/exp/yolov2-model"

labels="/datasets/BigLearning/ahjiang/bb/udacity-od-crowdai/Udacity_object_dataset/labels/car-label.txt"
output="data/images/tmp/"
model_prefix="data/models/exp/yolov2-model-car"
plot_prefix="data/plots/crowdai/car/"
input="/datasets/BigLearning/ahjiang/bb/npz/crowdai-car-test.npz"

step=5
n=74

for i in `seq 0 $step $n`;
do
    model_path=$model_prefix"-"$i"fr.h5"
    plot_path=$plot_prefix"-"$i".pdf"
    echo python -u run_test.py --num_frozen $i -c $labels -o $output --model_path $model_path --mode 1 -t $input
done
