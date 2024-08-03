# adapted from here
# https://gist.github.com/bonlime/4e0d236cf98cd5b15d977dfa03a63643
mkdir -p data/imagenet && cd data/imagenet
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
