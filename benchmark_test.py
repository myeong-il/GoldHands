import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import inception_v3, Inception3
from torchvision.utils import save_image
from torchvision import utils as vutils
from torch.utils.data import DataLoader

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import numpy as np
from scipy import linalg
from tqdm import tqdm
import pickle
import os
from utils import true_randperm
from datasets import InfiniteSamplerWrapper

# Inception weights ported to Pytorch from
# http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False,
                 use_fid_inception=False):
        """Build pretrained InceptionV3
        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradients. Possibly useful
            for finetuning the network
        use_fid_inception : bool
            If true, uses the pretrained Inception model used in Tensorflow's
            FID implementation. If false, uses the pretrained Inception model
            available in torchvision. The FID Inception model has different
            weights and a slightly different structure from torchvision's
            Inception model. If you want to compute FID scores, you are
            strongly advised to set this parameter to true to get comparable
            results.
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        if use_fid_inception:
            inception = fid_inception_v3()
        else:
            inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp




class Inception3Feature(Inception3):
    def forward(self, x):
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)

        x = self.Conv2d_1a_3x3(x)  # 299 x 299 x 3
        x = self.Conv2d_2a_3x3(x)  # 149 x 149 x 32
        x = self.Conv2d_2b_3x3(x)  # 147 x 147 x 32
        x = F.max_pool2d(x, kernel_size=3, stride=2)  # 147 x 147 x 64

        x = self.Conv2d_3b_1x1(x)  # 73 x 73 x 64
        x = self.Conv2d_4a_3x3(x)  # 73 x 73 x 80
        x = F.max_pool2d(x, kernel_size=3, stride=2)  # 71 x 71 x 192

        x = self.Mixed_5b(x)  # 35 x 35 x 192
        x = self.Mixed_5c(x)  # 35 x 35 x 256
        x = self.Mixed_5d(x)  # 35 x 35 x 288

        x = self.Mixed_6a(x)  # 35 x 35 x 288
        x = self.Mixed_6b(x)  # 17 x 17 x 768
        x = self.Mixed_6c(x)  # 17 x 17 x 768
        x = self.Mixed_6d(x)  # 17 x 17 x 768
        x = self.Mixed_6e(x)  # 17 x 17 x 768

        x = self.Mixed_7a(x)  # 17 x 17 x 768
        x = self.Mixed_7b(x)  # 8 x 8 x 1280
        x = self.Mixed_7c(x)  # 8 x 8 x 2048

        x = F.avg_pool2d(x, kernel_size=8)  # 8 x 8 x 2048

        return x.view(x.shape[0], x.shape[1])  # 1 x 1 x 2048


def load_patched_inception_v3():
    # inception = inception_v3(pretrained=True)
    # inception_feat = Inception3Feature()
    # inception_feat.load_state_dict(inception.state_dict())
    inception_feat = InceptionV3([3], normalize_input=False)

    return inception_feat


@torch.no_grad()
def extract_features(loader, inception, device):
    pbar = tqdm(loader)

    feature_list = []

    for img in pbar:
        img = img.to(device)
        feature = inception(img)[0].view(img.shape[0], -1)
        feature_list.append(feature.to('cpu'))

    features = torch.cat(feature_list, 0)

    return features



@torch.no_grad()
def extract_feature_from_samples(generator, inception, device='cuda'):
    n_batch = n_sample // batch_size
    resid = n_sample - (n_batch * batch_size)
    batch_sizes = [batch_size] * n_batch + [resid]
    features = []

    for batch in tqdm(batch_sizes):
        latent = torch.randn(batch, 512, device=device)
        img, _ = g([latent], truncation=truncation, truncation_latent=truncation_latent)
        feat = inception(img)[0].view(img.shape[0], -1)
        features.append(feat.to('cpu'))

    features = torch.cat(features, 0)

    return features


@torch.no_grad()
def extract_feature_from_generator_fn(generator_fn, inception, device='cuda', total=1000):
    features = []

    for batch in tqdm(generator_fn, total=total):
        try:
            feat = inception(batch)[0].view(batch.shape[0], -1)
            features.append(feat.to('cpu'))
        except:
            break
    features = torch.cat(features, 0).detach()
    return features.numpy()




def real_image_loader(dataloader, n_batches=10):
    counter = 0
    while counter < n_batches:
        counter += 1
        rgb_img = next(dataloader)[0]
        if counter == 1:
            vutils.save_image(0.5*(rgb_img+1), 'tmp_real.jpg')  
        yield rgb_img.cuda()




@torch.no_grad()
def image_generator(dataset, net_ae, net_ig, BATCH_SIZE=1, n_batches=500):
    counter = 0
    dataloader = iter(DataLoader(dataset, BATCH_SIZE, sampler=InfiniteSamplerWrapper(dataset), num_workers=4, pin_memory=False))
    n_batches = min( n_batches, len(dataset)//BATCH_SIZE-1 )
    while counter < n_batches:
        counter += 1
        rgb_img, _, _, skt_img = next(dataloader)
        rgb_img = F.interpolate( rgb_img, size=512 ).cuda()
        skt_img = F.interpolate( skt_img, size=512 ).cuda()

        gimg_ae, style_feat = net_ae(skt_img, rgb_img)
        g_image = net_ig(gimg_ae, style_feat)
        if counter == 1:
            vutils.save_image(0.5*(g_image+1), './static/predict/tmp.jpg')        
        yield g_image


@torch.no_grad()
def image_generator_perm(dataset, net_ae, net_ig, BATCH_SIZE=1, n_batches=500):
    counter = 0
    dataloader = iter(DataLoader(dataset, BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=False))
    n_batches = min( n_batches, len(dataset)//BATCH_SIZE-1 )
    while counter < n_batches:
        counter += 1
        rgb_img, _, _, skt_img = next(dataloader)
        rgb_img = F.interpolate( rgb_img, size=512 ).cuda()
        skt_img = F.interpolate( skt_img, size=512 ).cuda()

        perm = true_randperm(rgb_img.shape[0], device=rgb_img.device)

        gimg_ae, style_feat = net_ae(skt_img, rgb_img[perm])
        g_image = net_ig(gimg_ae, style_feat)
        if counter == 1:
            vutils.save_image(0.5*(g_image+1), 'tmp.jpg')        
        yield g_image


def mainfun(file_name):
    print('들어왔어요~')
    print(file_name)
    from datasets import PairedMultiDataset, InfiniteSamplerWrapper
    from utils import  make_folders ,AverageMeter
    from torch.utils.data import DataLoader
    from torchvision import utils as vutils
    IM_SIZE = 1024
    BATCH_SIZE = 1
    DATALOADER_WORKERS = 4
    NBR_CLS = 2000
    TRIAL_NAME = 'trial_vae_512_1'
    SAVE_FOLDER = './'

    data_root_colorful = './imgStyle/'+file_name
    data_root_sketch_1 = './static/uploads'
    data_root_sketch_2 = './static/uploads'
    data_root_sketch_3 = './static/uploads'

    dataset = PairedMultiDataset(data_root_colorful, data_root_sketch_1, data_root_sketch_2, data_root_sketch_3, im_size=IM_SIZE, rand_crop=False)
    dataloader = iter(DataLoader(dataset, BATCH_SIZE, shuffle=False, num_workers=DATALOADER_WORKERS, pin_memory=True))


    from models import StyleEncoder, ContentEncoder, Decoder
    import pickle
    from models import AE, RefineGenerator_face
    from utils import load_params

    net_ig = RefineGenerator_face(nfc=64, im_size=256).cuda()
    net_ig = nn.DataParallel(net_ig)

    ckpt = './train_results/GAN_trial-pr-face-8-20-23-30/models/19.pth'
    if ckpt is not None:
        ckpt = torch.load(ckpt)
        #net_ig.load_state_dict(ckpt['ig'])
        #net_id.load_state_dict(ckpt['id'])
        net_ig_ema = ckpt['ig_ema']
        load_params(net_ig, net_ig_ema)
    net_ig = net_ig.module
    #net_ig.eval()
    
    print('gan')

    net_ae = AE(nfc=64)
    net_ae.load_state_dicts('./train_results/AE_trial-pr-face-8-20-23-30/models/100000.pth')
    net_ae.cuda()
    net_ae.eval()
    
    print('ae')

    inception = load_patched_inception_v3().cuda()
    inception.eval()


    real_features = pickle.load( open('face_fid_feats.npy', 'rb') )
    real_mean = real_features['mean']
    real_cov = real_features['cov']

    for it in range(1):
        #itx = it * 8000
        '''
        ckpt = torch.load('./train_results/%s/models/%d.pth'%(TRIAL_NAME, itx))

        style_encoder.load_state_dict(ckpt['e'])
        content_encoder.load_state_dict(ckpt['c'])
        decoder.load_state_dict(ckpt['d'])`
        
        dataloader = iter(DataLoader(dataset, BATCH_SIZE, sampler=InfiniteSamplerWrapper(dataset), num_workers=DATALOADER_WORKERS, pin_memory=True))
        '''
        
        sample_features = extract_feature_from_generator_fn( 
            image_generator(dataset, net_ae, net_ig, n_batches=1800), inception,
             total=1800 )



# if __name__ == "__main__":
#     main()