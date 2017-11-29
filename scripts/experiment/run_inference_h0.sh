labels="/datasets/BigLearning/ahjiang/bb/udacity-od-crowdai/Udacity_object_dataset/labels/car-label.txt"
output="data/images/crowdai/car/"
model_prefix="data/models/exp/yolov2-model-car"
plot_prefix="data/plots/crowdai/car/pr"
input="/datasets/BigLearning/ahjiang/bb/npz/crowdai-car-test.npz"

step=5
n=74

for i in `seq 0 $step $n`;
do
    model_path=$model_prefix"-"$i"fr.h5"
    plot_path=$plot_prefix"-"$i".pdf"
    python -u run_test.py --num_frozen $i -p $plot_path -c $labels -o $output --model_path $model_path --mode 1 -t $input
done

#input="/datasets/BigLearning/ahjiang/bb/udacity-od-crowdai/object-detection-crowdai-scaled/images/test/"
#i=70
#model_path=$model_prefix"-"$i"fr.h5"
#plot_path=$plot_prefix"-"$i".pdf"
#python -u run_test.py -s 0.0001 --num_frozen $i -p $plot_path -c $labels -o $output --model_path $model_path --mode 0 -t $input

