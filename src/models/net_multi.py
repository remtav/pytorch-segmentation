from .common import ActivatedBatchNorm
from .encoder import create_encoder
from .decoder import create_decoder
from .spp import create_spp, create_mspp
from .tta import SegmentatorTTA
import torch.nn as nn
from collections import OrderedDict
from .efficientunet.layers import *

class EncoderDecoderNet(nn.Module, SegmentatorTTA):
    def __init__(self, output_channels=19, enc_type='resnet50', dec_type='unet_scse',
                 num_filters=16, pretrained=False):
        super().__init__()
        self.output_channels = output_channels
        self.enc_type = enc_type
        self.dec_type = dec_type

        assert enc_type in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                            'resnext101_32x4d', 'resnext101_64x4d',
                            'se_resnet50', 'se_resnet101', 'se_resnet152',
                            'se_resnext50_32x4d', 'se_resnext101_32x4d', 'senet154', 'efficientnet']
        assert dec_type in ['unet_scse', 'unet_seibn', 'unet_oc']

        encoder = create_encoder(enc_type, pretrained)
        Decoder = create_decoder(dec_type)

        self.encoder1 = encoder[0]
        self.encoder2 = encoder[1]
        self.encoder3 = encoder[2]
        self.encoder4 = encoder[3]
        self.encoder5 = encoder[4]

        self.pool = nn.MaxPool2d(2, 2)
        self.center = Decoder(self.encoder5.out_channels, num_filters * 32 * 2, num_filters * 32)

        self.decoder5 = Decoder(self.encoder5.out_channels + num_filters * 32, num_filters * 32 * 2,
                                num_filters * 16)
        self.decoder4 = Decoder(self.encoder4.out_channels + num_filters * 16, num_filters * 16 * 2,
                                num_filters * 8)
        self.decoder3 = Decoder(self.encoder3.out_channels + num_filters * 8, num_filters * 8 * 2, num_filters * 4)
        self.decoder2 = Decoder(self.encoder2.out_channels + num_filters * 4, num_filters * 4 * 2, num_filters * 2)
        self.decoder1 = Decoder(self.encoder1.out_channels + num_filters * 2, num_filters * 2 * 2, num_filters)

        self.logits = nn.Sequential(
            nn.Conv2d(num_filters * (16 + 8 + 4 + 2 + 1), 64, kernel_size=1, padding=0),
            ActivatedBatchNorm(64),
            nn.Conv2d(64, self.output_channels, kernel_size=1)
        )

    def forward(self, x):
        img_size = x.shape[2:]

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        c = self.center(self.pool(e5))
        e1_up = F.interpolate(e1, scale_factor=2, mode='bilinear', align_corners=False)

        #print([e5.size(), c.size()])

        d5 = self.decoder5(c, e5)
        d4 = self.decoder4(d5, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1_up) #TODO: vérifier résolution

        #u5 = F.interpolate(d5, img_size, mode='bilinear', align_corners=False)
        #u4 = F.interpolate(d4, img_size, mode='bilinear', align_corners=False)
        #u3 = F.interpolate(d3, img_size, mode='bilinear', align_corners=False)
        #u2 = F.interpolate(d2, img_size, mode='bilinear', align_corners=False)

        # Hyper column
        #d = torch.cat((d1, u2, u3, u4, u5), 1)

        logits = self.logits(d1)

        return logits


__all__ = ['EfficientUnet', 'get_efficientunet_b0', 'get_efficientunet_b1', 'get_efficientunet_b2',
           'get_efficientunet_b3', 'get_efficientunet_b4', 'get_efficientunet_b5', 'get_efficientunet_b6',
           'get_efficientunet_b7']

class EfficientUnet(nn.Module, SegmentatorTTA):
    def __init__(self, encoder, out_channels=2, concat_input=True, pretrained=True):
        super().__init__()

        self.encoder = create_encoder(encoder, pretrained=pretrained)
        self.concat_input = concat_input

        self.up_conv1 = up_conv(self.n_channels, 512)
        self.double_conv1 = double_conv(self.size[0], 512)
        self.up_conv2 = up_conv(512, 256)
        self.double_conv2 = double_conv(self.size[1], 256)
        self.up_conv3 = up_conv(256, 128)
        self.double_conv3 = double_conv(self.size[2], 128)
        self.up_conv4 = up_conv(128, 64)
        self.double_conv4 = double_conv(self.size[3], 64)

        if self.concat_input:
            self.up_conv_input = up_conv(64, 32)
            self.double_conv_input = double_conv(self.size[4], 32)

        self.final_conv = nn.Conv2d(self.size[5], out_channels, kernel_size=1)

    @property
    def n_channels(self):
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        return n_channels_dict[self.encoder.name]

    @property
    def size(self):
        size_dict = {'efficientnet-b0': [592, 296, 152, 80, 35, 32], 'efficientnet-b1': [592, 296, 152, 80, 35, 32],
                     'efficientnet-b2': [600, 304, 152, 80, 35, 32], 'efficientnet-b3': [608, 304, 160, 88, 35, 32],
                     'efficientnet-b4': [624, 312, 160, 88, 35, 32], 'efficientnet-b5': [640, 320, 168, 88, 35, 32],
                     'efficientnet-b6': [656, 328, 168, 96, 35, 32], 'efficientnet-b7': [672, 336, 176, 96, 35, 32]}
        return size_dict[self.encoder.name]

    def forward(self, x):
        input_ = x

        block10, block6, block2, block0, head_swish = self.encoder(x)
        x = head_swish

        x = self.up_conv1(x)
        x = torch.cat([x, block10], dim=1)  # blocks_10_output_batch_norm
        x = self.double_conv1(x)

        x = self.up_conv2(x)
        x = self.double_conv2(x)

        x = self.up_conv3(x)
        x = self.double_conv3(x)

        x = self.up_conv4(x)
        x = self.double_conv4(x)

        if self.concat_input:
            x = self.up_conv_input(x)
            x = torch.cat([x, input_], dim=1)
            x = self.double_conv_input(x)

        x = self.final_conv(x)

        return x

class SPPNetEncoder(nn.Module, SegmentatorTTA):
    #TODO: adapt to different number of tasks. For now, hard coded to 3 tasks
    #TODO: adapt to tasks with different output_channels. For now, hard coded for 2 out channels
    #TODO: implement mobilnetv2 as multitask
    def __init__(self, enc_type='xception65', output_stride=8, pretrained_path=False):
        super().__init__()
        self.enc_type = enc_type
        self.pretrained_path = pretrained_path

        assert enc_type in ['xception65', 'mobilenetv2']

        pretrained = True if not pretrained_path else False

        self.encoder = create_encoder(enc_type, output_stride=output_stride, pretrained=pretrained)

    #def load_weights(self, pretrained_path=False):
        #if not pretrained_path:
        #    print('Unable to load weights: no pretrained path was given.')
        #else:
        if self.pretrained_path:
            param = torch.load(self.pretrained_path)#, map_location='cpu')
            #pretrained_dict = param
            #model_dict = self.state_dict()
            self.load_state_dict(param, strict=False)
            del param

    def forward(self, inputs):
        if self.enc_type == 'mobilenetv2':
            x = self.encoder(inputs)
            return x
        else:
            x, low_level_feat = self.encoder(inputs)
            #x = self.spp(x)

            return x, low_level_feat

class SPPNetDecoder(nn.Module, SegmentatorTTA):
    #TODO: adapt to different number of tasks. For now, hard coded to 3 tasks
    #TODO: adapt to tasks with different output_channels. For now, hard coded for 2 out channels
    #TODO: implement mobilnetv2 as multitask
    def __init__(self, output_channels=19, enc_type='xception65', dec_type='aspp', output_stride=8, pretrained_path=False):
        super().__init__()
        self.output_channels = output_channels
        self.enc_type = enc_type
        self.dec_type = dec_type
        self.pretrained_path = pretrained_path

        assert enc_type in ['xception65', 'mobilenetv2']
        assert dec_type in ['oc_base', 'oc_asp', 'spp', 'aspp', 'maspp']

        #self.encoder = create_encoder(enc_type, output_stride=output_stride, pretrained=True)
        #self.encoder = create_encoder(enc_type, output_stride=output_stride, pretrained=False)
        if enc_type == 'mobilenetv2':
            #raise NotImplementedError
            self.spp = create_mspp(dec_type)
        else:
            self.spp, self.decoder = create_spp(dec_type, output_stride=output_stride)
        self.logits = nn.Conv2d(256, output_channels, 1)

        if self.pretrained_path:
            param = torch.load(self.pretrained_path)#, map_location='cpu')
            pretrained_dict = param
            model_dict = self.state_dict()

            if self.state_dict()['logits.weight'].shape != pretrained_dict['logits.weight'].shape:
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k.find('logits') == -1}
                # 2. overwrite entries in the existing state dict
                model_dict.update(pretrained_dict)
                # 3. load the new state dict
                self.load_state_dict(model_dict, strict=False)
            else:
                self.load_state_dict(param, strict=False)
            del param

    def forward(self, embedding, low_level_feat=None):
        if self.enc_type == 'mobilenetv2':
            #raise NotImplementedError
            x = self.spp(embedding)
            x = self.logits(x)
            return x
        else:
            #x, low_level_feat = self.encoder(inputs)
            embedding = self.spp(embedding)
            x = self.decoder(embedding, low_level_feat)
            x = self.logits(x)
            return x

class SPPNetMulti(nn.Module, SegmentatorTTA):
    def __init__(self, output_channels=19, enc_type='xception65', dec_type='aspp', output_stride=8, pretrained_path=False):
        super().__init__()
        self.output_channels = output_channels
        self.enc_type = enc_type
        self.dec_type = dec_type
        self.pretrained_path = pretrained_path

        assert enc_type in ['xception65', 'mobilenetv2']
        assert dec_type in ['oc_base', 'oc_asp', 'spp', 'aspp', 'maspp']

        self.encoder = SPPNetEncoder(enc_type, output_stride, pretrained_path)
        self.decoder_task1 = SPPNetDecoder(output_channels, enc_type, dec_type, output_stride, pretrained_path)
        self.decoder_task2 = SPPNetDecoder(output_channels, enc_type, dec_type, output_stride, pretrained_path)
        self.decoder_task3 = SPPNetDecoder(output_channels, enc_type, dec_type, output_stride, pretrained_path)

    def forward(self, inputs):
        if self.enc_type == 'mobilenetv2':
            #raise NotImplementedError
            x = self.encoder.encoder(inputs)
            x_task1 = self.decoder_task1(x)
            x_task2 = self.decoder_task2(x)
            x_task3 = self.decoder_task3(x)

        else:
            print('TODO: check that forward pass incorporates logits...')
            x, low_level_feat = self.encoder.encoder(inputs)
            x = self.decoder_task1.spp(x)
            x_task1 = self.decoder_task1.decoder(x, low_level_feat)
            x_task2 = self.decoder_task2.decoder(x, low_level_feat)
            x_task3 = self.decoder_task3.decoder(x, low_level_feat)

        return x_task1, x_task2, x_task3