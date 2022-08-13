import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.


    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        # configure config
        self.cfg = (coco, voc)[num_classes == 21]
        # Initialize a priori box
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        #backbone
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        # The network behind conv4_3, L2 regularization
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        # Regression and classification networks
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])



        if phase == 'test':
            self.softmax = nn.Softmax()
            self.detect = Detect()

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        # vgg network to conv4_3
        for k in range(23):
            x = self.vgg[k](x)

        # l2 regularization
        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        # conv4_3 to fc
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        # extras network
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                # Store the network output that needs to be multi-scaled into sources
                sources.append(x)

        # apply multibox head to source layers
        # Multiscale regression and classification networks
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect.apply(21, 0, 200, 0.01, 0.45,
                                       loc.view(loc.size(0), -1, 4),  # loc preds
                                       self.softmax(conf.view(-1,
                                                              21)),  # conf preds
                                       self.priors.type(type(x.data))  # default boxes
                                       )

        else:
            output = (
                # loc output, size: (batch, 8732, 4)
                loc.view(loc.size(0), -1, 4),
                # conf output, size: (batch, 8732, 21)
                conf.view(conf.size(0), -1, self.num_classes),
                # Generate all candidate boxes size([8732, 4])
                self.priors
            )
        return output

    # load model parameters
    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')



def vgg(cfg, i, batch_norm=False):
    '''
    The code refers to the code of vgg official website
    '''
    layers = []
    in_channels = i
    for v in cfg:
        # normal max_pooling
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        # ceil_mode = True, upsampling makes channel 75-->38
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            # update in_channels
            in_channels = v
    # max_pooling (3,3,1,1)
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    # Newly added network layer 1024x3x3
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    # Newly added network layer 1024x1x1
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    # Incorporate into the overall network
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    # Initial input channel is 1024
    in_channels = i
    # flag is used to select kernel_size= 1 or 3
    flag = False    #flag is used to control kernel_size= 1 or 3
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    '''
    Args:
         vgg: vgg network after modifying fc
         extra_layers: 4-layer network added after vgg
         cfg: network parameters, eg: [4, 6, 6, 6, 4, 4]
         num_classes: categories, VOC is 20+background=21
     Return:
         vgg, extra_layers
         loc_layers: regression network with multi-scale branches
         conf_layers: classification network with multi-scale branches
    '''

    loc_layers = [] #Multi-scale branch regression network
    conf_layers = []    #Multi-scale branch classification network
    # The first part, Conv2d-4_3 (21 layers) of vgg network, Conv2d-7_1 (-2 layers)
    vgg_source = [21, -2]
    # The first part, Conv2d-4_3 (21 layers) of vgg network, Conv2d-7_1 (-2 layers)
    for k, v in enumerate(vgg_source):
        # return box*4 (coordinates)
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        # Confidence box*(num_classes)
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    # In the second part, cfg starts from the third as the number of boxes, and the networks used for multi-scale extraction are 1, 3, 5, and 7 layers respectively
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    # Call multibox to generate vgg, extras, head
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes)
