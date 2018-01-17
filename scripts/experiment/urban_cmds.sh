
# sherbrooke 

labels="/datasets/BigLearning/ahjiang/bb/urban-tracker/labels/car-label.txt"
model_prefix="data/models/exp/yolov2-sherbrooke-car"
plot_prefix="data/plots/urban-tracker/sherbrooke/car/pr"
output="data/images/urban-tracker/sherbrooke/car"
input="/datasets/BigLearning/ahjiang/bb/npz/urban-tracker-sherbrooke-car-test.npz"

# stmarc

labels="/datasets/BigLearning/ahjiang/bb/urban-tracker/labels/car-label.txt"
model_prefix="data/models/exp/yolov2-stmarc-car"
plot_prefix="data/plots/urban-tracker/stmarc/car/pr"
output="data/images/urban-tracker/stmarc/car"
input="/datasets/BigLearning/ahjiang/bb/npz/urban-tracker-stmarc-car-test.npz"

# hybrid

labels="/datasets/BigLearning/ahjiang/bb/urban-tracker/labels/car-label.txt"
model_prefix="data/models/exp/yolov2-rouen-sherbrooke-stmarc-car"
output="data/images/urban-tracker/hybrid/car/"
plot_prefix="data/plots/urban-tracker/hybrid/car/pr"
input="/datasets/BigLearning/ahjiang/bb/npz/urban-tracker-rouen-sherbrooke-car-test.npz"

## Training

# Rouen

npz="/datasets/BigLearning/ahjiang/bb/npz/urban-tracker-rouen-car-training.npz"
labels="/datasets/BigLearning/ahjiang/bb/urban-tracker/labels/car-label.txt"
outmodel="data/models/exp/yolov2-rouen-car"

# CrowdAI

npz="/datasets/BigLearning/ahjiang/bb/npz/crowdai-car-training-0.8.npz"
labels="/datasets/BigLearning/ahjiang/bb/udacity-od-crowdai/Udacity_object_dataset/labels/car-label.txt"
outmodel="data/models/exp/yolov2-model-car-yolo2"
