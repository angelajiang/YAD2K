input="/users/ahjiang/src/YAD2K/data/images/crowdai/car/10/"
output="/users/ahjiang/Videos/bb/crowdai-car-s0.8-fr10.mp4"
ffmpeg -framerate 10 -pattern_type glob -i $input"*.jpg" -c:v libx264 -pix_fmt yuv420p $output
