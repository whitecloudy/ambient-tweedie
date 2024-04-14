mkdir -p $BASE_PATH/datasets/  # create datasets folder if it does not exist
cd $BASE_PATH/datasets  # move to datasets folder


# ------- LAION-10k subset --------- #
gdown --id 1lUUzhq9yK1YOWJePm12GeJgsUtCfpm_o  # images
gdown --id 1svzxtGgJRtCyWNW6pQPNGSDkhuDifTzn  # captions
unzip train.zip # unzip LAION images
mkdir -p laion-10k # create dir
mv train/images_large/* laion-10k/ # move to appropriate folder
mv 
rm -rf train/ # clean files
rm -rf train.zip # clean files


# ------- CelebA-HQ --------- #
mkdir -p celeba_hq
mkdir -p download_scripts
cd download_scripts
wget https://github.com/clovaai/stargan-v2/raw/master/download.sh

bash download.sh celeba-hq-dataset
mkdir -p celeba_hq_train_split
mv data/celeba_hq/train/male/* celeba_hq_train_split/
mv data/celeba_hq/train/female/* celeba_hq_train_split/
mkdir -p celeba_hq_eval_split
mv data/celeba_hq/val/male/* celeba_hq_eval_split/
mv data/celeba_hq/val/female/* celeba_hq_eval_split/
rm -rf data/
mv celeba_hq_train_split ../celeba_hq/
mv celeba_hq_eval_split ../celeba_hq
cd ..


# ------- FFHQ --------- #
# download ffhq from huggingface
wget https://huggingface.co/datasets/yangtao9009/FFHQ1024/resolve/main/FFHQ-1024-1.zip?download=true
wget https://huggingface.co/datasets/yangtao9009/FFHQ1024/resolve/main/FFHQ-1024-2.zip?download=true


