labels="/datasets/BigLearning/ahjiang/bb/udacity-od-crowdai/Udacity_object_dataset/labels/crowdai-labels.txt"
output="data/images/tmp/"
model_prefix="data/models/exp/yolov2-model"
input="/datasets/BigLearning/ahjiang/bb/npz/crowdai-test.npz"

step=5
n=74

step2=0.05
n2=0.5

for i in `seq 0 $step $n`;
do
    for score in `seq 0.05 $step2 $n2`;
    do
        model_path=$model_prefix"-"$i"fr.h5"
        #echo $model_path
        #/users/ahjiang/src/yad2k/data/models/exp/yolov2-model-hybrid-0fr.h5
        python -u run_test.py --num_frozen $i -c $labels -s $score -o $output --model_path $model_path --mode 1 -t $input
    done
done
