# Directory Structure
high-res-stereo
├─ configs					# Training & Inference config argument
├─ dataloader
︳　 ├─ Carlaloader.py			# Load Carla dataset
︳　 ├─ KITTIloader2012.py			# Load KITTI2012 dataset
︳　 ├─ KITTIloader2015.py			# Load KITTI2015 dataset
︳　 ├─ Listfiles.py				# Load inference data
︳　 └─ MiddleburyLoader.py			# Training dataset preprocess
├─ dataset					# Dataset
├─ models
︳　 ├─ hsm.py				# Our network Architecture
︳　 └─ utils.py				# HSM, SegNet Architecture
├─ utils
├─ weights					# Our network pretrained weight
├─ requirements.txt
├─ submission.py				# Inference script
└─ train.py					# Training script

# Environment setup
Python: 3.7, Pytorch: 1.9.1, CUDA: 11.1

# Requirements
  tensorflow-gpu==1.15
tensorboardX>=1.4
networkx==2.3
scipy==1.2
opencv-python

# Create environment
conda create -n HR-stereo python=3.7 anaconda
conda activate HR-stereo
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

# Train
conda activate HR-stereo
python train.py --config './configs/train_argument.yml'
# train_argument.yml parameter introduction
--maxdisp			# maxium disparity
--logname			# log name
--database			# dataset path
--epochs			# number of epochs to train
--batchsize			# samples per batch
--loadmodel			# pretrained model path
--savemodel			# save model path

# Training Dataset Directory Structure
all_dataset
├─ dataset1
︳　 ├─ left images		# left image files
︳　 ├─ right images		# right image files
︳　 ├─ left semantic		# left semantic files
︳　 ├─ right semantic	# right semantic files
︳　 ├─ left semantic		# left instance files
︳　 ├─ right instance	# right instance files
︳　 └─ disp			# Disparity ground truth
└─ dataset2

# Inference
conda activate HR-stereo
python submission.py --config './configs/inference_argument.yml'
# inference_argument.yml parameter introduction
--datapath			# test data path
--loadmodel			# pretrained model path
--outdir			# output direction
--clean				# clean up output using entropy estimation
--testres			# test time resolution ratio
--max_disp			# maximum disparity to search for

# Inference Directory Structure
inference directory
├─ scence_0
︳　 ├─ im0.png		#left image
︳　 ├─ im1.png		# right image
︳　 ├─ im2.png		# left semantic
︳　 ├─ im3.png		# right semantic
︳　 ├─ im4.npy		# left instance
︳　 └─ im5.npy		# right instance
┋
┋
└─ scence_N
 
# Evaluation
conda activate HR-stereo
python eval.py --evalpath ‘./eval’ --GTpath ‘./data-my’
Hyperparameter
-- evalpath			# evaluation data path
-- GTpath			# ground truth path
