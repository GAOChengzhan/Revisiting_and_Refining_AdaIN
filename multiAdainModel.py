import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
import matplotlib.pyplot as plt

def calc_mean_std(features):
    """

    :param features: shape of features -> [batch_size, c, h, w]
    :return: features_mean, feature_s: shape of mean/std ->[batch_size, c, 1, 1]
    """

    batch_size, c = features.size()[:2]
    features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
    return features_mean, features_std


def adain(content_features, style_features):
    """
    Adaptive Instance Normalization

    :param content_features: shape -> [batch_size, c, h, w]
    :param style_features: shape -> [batch_size, c, h, w]
    :return: normalized_features shape -> [batch_size, c, h, w]
    """
    content_mean, content_std = calc_mean_std(content_features)
    style_mean, style_std = calc_mean_std(style_features)
    normalized_features = style_std * (content_features - content_mean) / content_std + style_mean
    return normalized_features


class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=True).features
        self.slice1 = vgg[: 2]
        self.slice2 = vgg[2: 7]
        self.slice3 = vgg[7: 12]
        self.slice4 = vgg[12: 21]
        for p in self.parameters():
            p.requires_grad = False

    def save_features(self, features_list, stage):
        for i, features in enumerate(features_list):
            # Select the first feature map of the first image
            feature = features[0][0].cpu().data.numpy()
            plt.imshow(feature, cmap='gray')
            plt.title(f"{stage} stage {i+1}")
            plt.colorbar()
            plt.savefig(f"{stage}_stage_{i+1}.png")
            plt.close()



    def forward(self, images, output_last_feature=False, save_features=False):
        h1 = self.slice1(images)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        
        if save_features:
            self.save_features([h1, h2, h3, h4], "encoder")
        
        if output_last_feature:
            return h4
        else:
            return h1,h2,h3,h4


class RC(nn.Module):
    """A wrapper of ReflectionPad2d and Conv2d"""
    def __init__(self, in_channels, out_channels, kernel_size=3, pad_size=1, activated=True):
        super().__init__()
        self.pad = nn.ReflectionPad2d((pad_size, pad_size, pad_size, pad_size))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.activated = activated

    def forward(self, x):
        h = self.pad(x)
        h = self.conv(h)
        if self.activated:
            return F.relu(h)
        else:
            return h


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rc1 = RC(512, 256, 3, 1)
        self.rc2 = RC(256*2, 256, 3, 1) # adjust for extra channels from t3
        self.rc3 = RC(256, 256, 3, 1)
        self.rc4 = RC(256, 128, 3, 1)
        self.rc5 = RC(128*2, 128, 3, 1) # adjust for extra channels from t2
        self.rc6 = RC(128, 128, 3, 1)
        self.rc7 = RC(128, 64, 3, 1)
        self.rc8 = RC(64, 64, 3, 1)
        self.rc9 = RC(64, 3, 3, 1, False)

    def forward(self, t2, t3, t4):
        h = self.rc1(t4)
        t3 = F.interpolate(t3, size=h.size()[2:], mode='nearest') # upsample t3 to match h
        h = self.rc2(torch.cat([h, t3], dim=1)) # combine with t3
        h = F.interpolate(h, scale_factor=2)
        h = self.rc3(h)
        h = self.rc4(h)
        t2 = F.interpolate(t2, size=h.size()[2:], mode='nearest') # upsample t2 to match h
        h = self.rc5(torch.cat([h, t2], dim=1)) # combine with t2
        h = F.interpolate(h, scale_factor=2)
        h = self.rc6(h)
        h = self.rc7(h)
        h = F.interpolate(h, scale_factor=2)
        h = self.rc8(h)
        h = self.rc9(h)
        return h



class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg_encoder = VGGEncoder()
        self.decoder = Decoder()

    def generate(self, content_images, style_images, alpha_lst=[1.0,1.0,1.0]):
        _,contentFeature_2,contentFeature_3,contentFeature_4 = self.vgg_encoder(content_images, output_last_feature=False)
        _,styleFeature_2,styleFeature_3,styleFeature_4  = self.vgg_encoder(style_images, output_last_feature=False)
        t2,t3,t4 = adain(contentFeature_2, styleFeature_2),\
                    adain(contentFeature_3, styleFeature_3),\
                    adain(contentFeature_4, styleFeature_4)
        t2 = t2*alpha_lst[0]+(1-alpha_lst[0])*contentFeature_2
        t3 = t3*alpha_lst[1]+(1-alpha_lst[1])*contentFeature_3
        t4 = t4*alpha_lst[2]+(1-alpha_lst[2])*contentFeature_4
        out = self.decoder(t2,t3,t4)
        return out

    @staticmethod
    def calc_content_loss(out_features, t):
        return F.mse_loss(out_features, t)

    @staticmethod
    def calc_style_loss(content_middle_features, style_middle_features):
        loss = 0
        for c, s in zip(content_middle_features, style_middle_features):
            c_mean, c_std = calc_mean_std(c)
            s_mean, s_std = calc_mean_std(s)
            loss += F.mse_loss(c_mean, s_mean) + F.mse_loss(c_std, s_std)
        return loss

    def forward(self, content_images, style_images, alpha=1.0, lam=10):
        _,contentFeature_2,contentFeature_3,contentFeature_4 = self.vgg_encoder(content_images, output_last_feature=False)
        _,styleFeature_2,styleFeature_3,styleFeature_4  = self.vgg_encoder(style_images, output_last_feature=False)
        t2,t3,t4 = adain(contentFeature_2, styleFeature_2),\
                    adain(contentFeature_3, styleFeature_3),\
                    adain(contentFeature_4, styleFeature_4)
        
        out = self.decoder(t2,t3,t4)

        _,out2,out3,out4 = self.vgg_encoder(out, output_last_feature=False)
        output_middle_features = self.vgg_encoder(out, output_last_feature=False)
        style_middle_features = self.vgg_encoder(style_images, output_last_feature=False)
        loss_c=0
        for out,t in zip([out2,out3,out4],[t2,t3,t4]):
            loss_c += self.calc_content_loss(out, t)

        loss_s = self.calc_style_loss(output_middle_features, style_middle_features)
        loss = loss_c + lam * loss_s
        return loss,loss_c,loss_s
