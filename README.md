# CodeChangeAI

This project aims to help us analysis the change log of qemu new releases automatically and tells the testers which features are changed and need to be tested.

# Usage
usage example:

training:

time python src/changelog.py data/qemu*/*.docx

the input is paths to all changelog/docx files. the file names will be the ground truth label for all commits in that file. if the tag names change, you need to rename the files accordingly and train again. a csv file called data.csv will be generated.

time python src/multi*.py 

train a model on the csv file generated from the previous command. 

predicting:
(pytorch) [root@xxx demo]# time python src/infer.py data/changelog-0819 

the input can be paths to changelogs. this program can take multiple inputs and will generate predictions for each input file separately.

# File Descriptions
data/

generated files:

data.csv is an intermediate file with the commits and labels. but it will be overwritten by the infer code.

classification_report.txt  has the precision and recall on the validation set.

> The Area Under the Curve (AUC) is the measure of the ability of a classifier to distinguish between classes and is used as a summary of the ROC curve. The higher the AUC, the better the performance of the model at distinguishing between the positive and negative classes.

textcnn_model is the trained model.

Train Validation Loss can be used to detect overfitting.

word_to_tensor.pt converts words in commits to tensors that can be fed into the model.

# Setup
install an Anaconda environment. i used miniconda since it's lightweight. 'use pip only after conda'. pytorch needs to be installed after cuda.

conda packages are installed from different channels. for example,
conda install -c bjrn pandarallel 
conda install -c conda-forge python-docx 

https://mirror.tuna.tsinghua.edu.cn/help/anaconda/ 

# Cuda 10.2
https://docs.nvidia.com/cuda/archive/10.2/cuda-installation-guide-linux/index.html

Disable the Nouveau drivers. many steps on this link have to be modified according to the cuda version. yum search cuda shows the available version. for example,
yum install cuda-10-2.x86_64. .

export PATH=/usr/local/cuda-10.2/bin:/opt/nvidia/nsight-compute/${PATH:+:${PATH}}

the runfile can be found at the below link
https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Linux&target_arch=x86_64&target_distro=RHEL&target_version=8&target_type=runfilelocal 
https://developer.nvidia.com/rdp/cudnn-archive
https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.2.1.32/10.2_06072021/RHEL8_1-x64/libcudnn8-8.2.1.32-1.cuda10.2.x86_64.rpm

https://pytorch.org/get-started/locally/
https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#download

CUDA SDK 10.0 – 10.2 support for compute capability 3.0 – 7.5 

the default pytorch only supports GPU Compute Capability
'sm_37', 'sm_50', 'sm_60', 'sm_61', 'sm_70', 'sm_75',

in order to use other gpu models you need to build pytorch from source.
https://github.com/pytorch/pytorch#from-source

therefore, pytorch version in the requirements file have to be changed according to the hardware.

# References:

https://github.com/diardanoraihan/Text_Classification_Capstone/blob/main/3_WordEmbedding_Models/Word2Vec/1_CNN/CNN_CR.ipynb

https://nbviewer.jupyter.org/github/MLWhiz/data_science_blogs/blob/a494cf5cbb5ae304926e14fb5a9e5421b6409910/multiclass/multiclass-text-classification-pytorch.ipynb

https://github.com/ahmedbesbes/multi-label-sentiment-classifier/blob/main/training_with_tez.ipynb

https://curiousily.com/posts/multi-label-text-classification-with-bert-and-pytorch-lightning/
