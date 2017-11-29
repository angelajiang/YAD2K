#ffmpeg -framerate 10 -pattern_type glob -i "/users/ahjiang/src/YAD2K/data/images/crowdai/truck/*.jpg" -c:v libx264 -pix_fmt yuv420p ~/Videos/bb/crowdai-truck-s0.0001-fr5.mp4
#ffmpeg -framerate 10 -pattern_type glob -i "/users/ahjiang/src/YAD2K/data/images/crowdai/car/*.jpg" -c:v libx264 -pix_fmt yuv420p ~/Videos/bb/crowdai-car-s0.3-fr5.mp4
ffmpeg -framerate 10 -pattern_type glob -i "/users/ahjiang/src/YAD2K/data/images/crowdai/pedestrian/*.jpg" -c:v libx264 -pix_fmt yuv420p ~/Videos/bb/crowdai-pedestrian-s0.0001-fr70.mp4
#ffmpeg -framerate 10 -pattern_type glob -i "/datasets/BigLearning/ahjiang/bb/udacity-od-crowdai/object-detection-crowdai-scaled/images/test/*.jpg" -c:v libx264 -pix_fmt yuv420p ~/Videos/bb/crowdai-test.mp4
#ffmpeg -framerate 10 -pattern_type glob -i "/datasets/BigLearning/ahjiang/bb/udacity-od-crowdai/object-detection-crowdai-scaled/images/training/*.jpg" -c:v libx264 -pix_fmt yuv420p ~/Videos/bb/crowdai-train.mp4
