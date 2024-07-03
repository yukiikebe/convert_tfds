xhost local:root

# docker run -it --rm --name tfds \
#     -v $PWD:/workspace \
#     yuki/tfds:tensorflow bash

docker run -it --rm -v $PWD:/workspace --gpus all yuki/tfds:tensorflow