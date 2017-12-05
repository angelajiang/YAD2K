
labels="/datasets/BigLearning/ahjiang/bb/udacity-od-crowdai/Udacity_object_dataset/labels/pedestrian-label.txt"
output="data/images/crowdai/pedestrian-augmented-2/"
model_prefix="data/models/exp/yolov2-model-pedestrian-augmented-2"
plot_prefix="data/plots/crowdai/pedestrian-augmented/pr2-"
input="/datasets/BigLearning/ahjiang/bb/npz/crowdai-pedestrian-test.npz"

start=0
step=5
end=74

for i in `seq $start $step $end`;
do
    model_path=$model_prefix"-"$i"fr.h5"
    plot_path=$plot_prefix"-"$i".pdf"
    python -u run_test.py --num_frozen $i -p $plot_path -c $labels -o $output --model_path $model_path --mode 1 -t $input
done

input="/datasets/BigLearning/ahjiang/bb/udacity-od-crowdai/object-detection-crowdai-scaled/images/test/"
#input="/datasets/BigLearning/ahjiang/bb/udacity-od-crowdai/object-detection-crowdai-scaled/images/training/"
i=20
model_path=$model_prefix"-"$i"fr.h5"
plot_path=$plot_prefix"-"$i".pdf"
#python -u run_test.py -s 0.3 --num_frozen $i -p $plot_path -c $labels -o $output --model_path $model_path --mode 0 -t $input
