import datetime


DATA_NAME = 'face'

DATALOADER_WORKERS = 2 #8
NBR_CLS = 500

EPOCH_GAN = 20

SAVE_IMAGE_INTERVAL = 500
SAVE_MODEL_INTERVAL = 2000
LOG_INTERVAL = 100
FID_INTERVAL = 2000
FID_BATCH_NBR = 500

ITERATION_AE = 100000

NFC=64
MULTI_GPU = False


IM_SIZE_GAN = 512 #1024
BATCH_SIZE_GAN = 16 #16

IM_SIZE_AE = 512
BATCH_SIZE_AE = 8 #16 #32

ct = datetime.datetime.now()  
TRIAL_NAME = 'trial-pr-face-10-04-17-05'# 'trial-pr-%s-%d-%d-%d-%d'%(DATA_NAME, ct.month, ct.day, ct.hour, ct.minute)
SAVE_FOLDER = './'

PRETRAINED_AE_PATH = './train_results/AE_trial-pr-face-10-02-15-07'  #None # 'add/the/pre-trained/model/path/if/fintuning'
PRETRAINED_AE_ITER = 100000

GAN_CKECKPOINT =None

TRAIN_AE_ONLY = False
TRAIN_GAN_ONLY = False

data_root_colorful = 'D:\\Jupyter\\model2\\dataset\\allcrop\\crop'  #'/path/to/image/folder'
data_root_sketch_1 = 'D:\\Jupyter\\model2\\dataset\\all_sketch\\s1'  #'/path/to/sketch/folder'
data_root_sketch_2 = 'D:\\Jupyter\\model2\\dataset\\all_sketch\\s2' #'/path/to/sketch/folder'
data_root_sketch_3 = 'D:\\Jupyter\\model2\\dataset\\all_sketch\\s3' #'/path/to/sketch/folder'
