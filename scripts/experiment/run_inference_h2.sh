#labels="/datasets/BigLearning/ahjiang/bb/udacity-od-crowdai/Udacity_object_dataset/labels/crowdai-labels.txt"
#output="data/images/tmp/"
#model_prefix="data/models/exp/yolov2-model"
#input="/datasets/BigLearning/ahjiang/bb/npz/crowdai-test.npz"

labels="/datasets/BigLearning/ahjiang/bb/udacity-od-crowdai/Udacity_object_dataset/labels/car-label.txt"
output="data/images/crowdai/car/"
model_prefix="data/models/exp/yolov2-model-car"
plot_prefix="data/plots/crowdai/car/pr"
input="/datasets/BigLearning/ahjiang/bb/npz/crowdai-car-test.npz"

step=5
start=0
end=74

for i in `seq $start $step $end`;
do
    model_path=$model_prefix"-"$i"fr.h5"
    plot_path=$plot_prefix"-"$i".pdf"
    #python -u run_test.py --num_frozen $i -p $plot_path -c $labels -o $output --model_path $model_path --mode 1 -t $input
done

# Make videos

labels="/datasets/BigLearning/ahjiang/bb/udacity-od-crowdai/Udacity_object_dataset/labels/car-label.txt"
output="data/images/crowdai/car/"
model_prefix="data/models/exp/yolov2-model-car"
input="/datasets/BigLearning/ahjiang/bb/udacity-od-crowdai/object-detection-crowdai-scaled/images/test/"

i=10
model_path=$model_prefix"-"$i"fr.h5"
plot_path=$plot_prefix"-"$i".pdf"
output_dir=$output"/"$i
mkdir $output_dir
rm -rf $output_dir"/*"
python -u run_test.py --num_frozen $i -s 0.8 -p $plot_path -c $labels -o $output_dir --model_path $model_path --mode 0 -t $input

labels="/datasets/BigLearning/ahjiang/bb/udacity-od-crowdai/Udacity_object_dataset/labels/pedestrian-label.txt"
output="data/images/crowdai/pedestrian/"
model_prefix="data/models/exp/yolov2-model-pedestrian-augmented-2"

