import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as sp_norm
import copy
import numpy as np
from .utils import to_rgb, from_rgb, to_decision, get_norm_by_name, print_model_parameters
from .feature_augmentation import Content_FA, Layout_FA
import matplotlib.pyplot as plt

def create_models(opt, recommended_config):
    """
    Build the model configurations and create models
    """
    config_G, config_D = prepare_config(opt, recommended_config)

    # --- generator and EMA --- #
    netG = Generator(config_G, opt.debug_folds).to(opt.device)
    netG.apply(weights_init)
    netEMA = copy.deepcopy(netG) if opt.use_EMA else None

    # --- discriminator --- #
    if opt.phase == "train":
        netD = Discriminator(config_D,  opt.debug_folds).to(opt.device)
        netD.apply(weights_init)
    else:
        netD = None

    # --- summary network of generator and discriminator--- #
    if opt.phase == "train":
        print("\nGenerator summary:\n")
        print_model_parameters(netG)
        
        print("\nDiscriminator summary:\n")
        print_model_parameters(netD)

    # --- load previous ckpt  --- #
    path = os.path.join(opt.checkpoints_dir, opt.exp_name, "models")
    if opt.continue_train or opt.phase == "test":
        netG.load_state_dict(torch.load(os.path.join(path, str(opt.continue_epoch).zfill(8)+"_G.pth")))
        print("Loaded Generator checkpoint")
        if opt.use_EMA:
            netEMA.load_state_dict(torch.load(os.path.join(path, str(opt.continue_epoch).zfill(8)+"_G_EMA.pth")))
            print("Loaded Generator_EMA checkpoint")
    if opt.continue_train and opt.phase == "train":
        netD.load_state_dict(torch.load(os.path.join(path, str(opt.continue_epoch).zfill(8)+"_D.pth")))
        print("Loaded Discriminator checkpoint")
    return netG, netD, netEMA


def create_optimizers(netG, netD, opt):
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, opt.beta2))
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, opt.beta2))
    return optimizerG, optimizerD


def prepare_config(opt, recommended_config):
    """
    Create model configuration dicts based on recommended settings and input parameters.
    Recommended num_blocks_d and num_blocks_d0 can be overridden by user inputs
    """
    G_keys_recommended = ['noise_shape', 'num_blocks_g', "use_masks", "num_mask_channels"]
    D_keys_recommended = ['num_blocks_d', 'num_blocks_d0', "use_masks", "num_mask_channels"]
    G_keys_user = ["ch_G", "norm_G", "noise_dim"]
    D_keys_user = ["ch_D", "norm_D", "prob_FA_con", "prob_FA_lay", "bernoulli_warmup"]

    config_G = dict((k, recommended_config[k]) for k in G_keys_recommended)
    config_G.update(dict((k, getattr(opt, k)) for k in G_keys_user))
    config_D = dict((k, recommended_config[k]) for k in D_keys_recommended)
    config_D.update(dict((k, getattr(opt, k)) for k in D_keys_user))

    if opt.num_blocks_d > 0:
        config_D["num_blocks_d"] = opt.num_blocks_d
    if opt.num_blocks_d0 > 0:
        config_D["num_blocks_d0"] = opt.num_blocks_d0
    return config_G, config_D



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_channels(which_net, base_multipler):
    channel_multipliers = {
        "Generator": [8, 8, 8, 8, 8, 8, 8, 4, 2, 1],
        "Discriminator": [1, 2, 4, 8, 8, 8, 8, 8, 8]
    }
    ans = list()
    for item in channel_multipliers[which_net]:
        ans.append(int(item * base_multipler))
    return ans


class Generator(nn.Module):
    def __init__(self, config_G, save_path=None):
        super(Generator, self).__init__()
        self.num_blocks        = config_G["num_blocks_g"]
        self.noise_shape       = config_G["noise_shape"]
        self.noise_init_dim    = config_G["noise_dim"]
        self.norm_name         = config_G["norm_G"]
        self.use_masks         = config_G["use_masks"]
        self.num_mask_channels = config_G["num_mask_channels"]
        self.save_path         = save_path
        num_of_channels = get_channels("Generator", config_G["ch_G"])[-self.num_blocks-1:]

        self.body, self.rgb_converters = nn.ModuleList([]), nn.ModuleList([])
        self.first_linear = nn.ConvTranspose2d(self.noise_init_dim, num_of_channels[0], self.noise_shape)
        for i in range(self.num_blocks):
            cur_block = G_block(num_of_channels[i], num_of_channels[i+1], self.norm_name, i==0)
            cur_rgb   = to_rgb(num_of_channels[i+1])
            self.body.append(cur_block)
            self.rgb_converters.append(cur_rgb)
        if self.use_masks:
            self.mask_converter = nn.Conv2d(num_of_channels[i+1], self.num_mask_channels, 3, padding=1, bias=True)

        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)

        print("Created Generator with %d parameters" % (sum(p.numel() for p in self.parameters())))

    def generate(self, z, get_feat=False, init_x = None, epoch=0):
        output     = dict()
        ans_images = list()
        ans_feat   = list()
        x = self.first_linear(z)  # 映射层

        if init_x is not None:
            x = x + 0.25 * init_x

        # ----------------------------------------------
        for i in range(self.num_blocks):
            x  = self.body[i](x)                       # 特征 
            im = torch.tanh(self.rgb_converters[i](x)) # 图像
            ans_images.append(im)
            ans_feat.append(torch.tanh(x))
        output["images"] = ans_images

        # ----------------------------------------------
        if get_feat:
             output["features"] = ans_feat
        
        # ----------------------------------------------
        if self.use_masks:
            mask = self.mask_converter(x)
            mask = F.softmax(mask, dim=1)
            output["masks"] = mask
        else:
            mask = None

        # --------- debug for images 、 features 、 mask --------- # 
        self.debug_images_features_mask(ans_images, ans_feat, mask, epoch)

        return output
    
    def debug_images_features_mask(self, ans_images, ans_feat, mask, epoch):
        if epoch % 500 == 0:
            plt.figure(figsize=(16, 24))
            cols_num = len(ans_images)
            row_num  = ans_images[0].shape[0] 
            index    = 0
            for i in range(row_num):
                # mask
                show_mask = mask.detach().cpu().numpy().squeeze() if mask is not None else None
                for j in range(cols_num):
                    # image
                    show_image = ans_images[j][i].detach().cpu().numpy().transpose(1, 2, 0)
                    show_image = show_image * 0.5 + 0.5

                    # feature
                    cur_feature = ans_feat[j][i].detach().cpu().numpy()
                    cur_feature = np.mean(cur_feature, axis=(0))
                    index       = i * 2 *(cols_num + 1) + j + 1
                    plt.subplot(row_num*2, cols_num + 1, index)
                    plt.imshow(cur_feature)
                    plt.axis('off')

                    cur_show = show_image
                    index       = (i * 2 + 1)*(cols_num + 1) + j + 1
                    plt.subplot(row_num*2, cols_num + 1, index)
                    plt.imshow(cur_show)
                    plt.axis('off')

                if show_mask is not None:
                    cur_mask = show_mask[i]
                    if len(cur_mask.shape) == 3 and cur_mask.shape[0] == 2:
                        cur_mask = np.transpose(cur_mask, (1, 2, 0))
                        cur_mask = np.concatenate((cur_mask, np.zeros_like(cur_mask[:, :, 0:1])), axis=2)

                    index    = (i * 2 + 1)*(cols_num + 1) + cols_num + 1
                    plt.subplot(row_num*2, cols_num + 1, index)
                    plt.imshow(cur_mask)
                    plt.axis('off')
            #plt.show()
            plt.suptitle('Fake_Image-Feature-Mask_Epoch_' + str(epoch))
            plt.tight_layout()
            save_path = 'Fake_Feature_Epoch_' + str(epoch).zfill(8) + '.png'
            save_path = os.path.join(self.save_path, save_path) if self.save_path else save_path
            plt.savefig(save_path)
            plt.close()


class G_block(nn.Module):
    def __init__(self, in_channel, out_channel, norm_name, is_first):
        super(G_block, self).__init__()
        middle_channel = min(in_channel, out_channel)
        self.ups   = nn.Upsample(scale_factor=2) if not is_first else torch.nn.Identity()
        self.activ = nn.LeakyReLU(0.2)
        self.conv1 = sp_norm(nn.Conv2d(in_channel,  middle_channel, 3, padding=1))
        self.conv2 = sp_norm(nn.Conv2d(middle_channel, out_channel, 3, padding=1))
        self.norm1 = get_norm_by_name(norm_name, in_channel)
        self.norm2 = get_norm_by_name(norm_name, middle_channel)
        self.conv_sc = sp_norm(nn.Conv2d(in_channel, out_channel, (1, 1), bias=False))

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x = self.activ(x)
        x = self.ups(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activ(x)
        x = self.conv2(x)
        h = self.ups(h)
        h = self.conv_sc(h)
        return h + x


class Discriminator(nn.Module):
    def __init__(self, config_D, save_path=None):
        super(Discriminator, self).__init__()
        self.num_blocks        = config_D["num_blocks_d"]
        self.num_blocks_ll     = config_D["num_blocks_d0"]
        self.norm_name         = config_D["norm_D"]
        self.prob_FA           = {"content": config_D["prob_FA_con"], "layout": config_D["prob_FA_lay"]}
        self.use_masks         = config_D["use_masks"]
        self.num_mask_channels = config_D["num_mask_channels"]
        self.bernoulli_warmup  = config_D["bernoulli_warmup"]
        self.save_path         = save_path
        num_of_channels        = get_channels("Discriminator", config_D["ch_D"])[:self.num_blocks + 1]
        
        if self.use_masks:
            for i in range(self.num_blocks_ll+1, self.num_blocks):
                num_of_channels[i] = int(num_of_channels[i] * 2)

        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)

        self.feature_prev_ratio = 8  # for msg concatenation

        self.body_ll,  self.body_content,  self.body_layout  = nn.ModuleList([]), nn.ModuleList([]), nn.ModuleList([])
        self.final_ll, self.final_content, self.final_layout = nn.ModuleList([]), nn.ModuleList([]), nn.ModuleList([])
        self.rgb_to_features = nn.ModuleList([])  # for msg concatenation

        # --------- D low-level --------- #
        for i in range(self.num_blocks_ll):
            msg_channels = num_of_channels[i] // self.feature_prev_ratio if i > 0 else num_of_channels[0]
            in_channels  = num_of_channels[i] + msg_channels if i > 0 else num_of_channels[0]
            cur_block    = D_block(in_channels, num_of_channels[i+1], self.norm_name, is_first=i == 0)
            self.body_ll.append(cur_block)
            self.rgb_to_features.append(from_rgb(msg_channels))
            self.final_ll.append(to_decision(num_of_channels[i+1], 1))

        # --------- D content --------- #
        self.content_FA = Content_FA(self.use_masks, self.prob_FA["content"], self.num_mask_channels, self.save_path)
        for i in range(self.num_blocks_ll, self.num_blocks):
            k = i - self.num_blocks_ll
            cur_block_content = D_block(num_of_channels[i], num_of_channels[i + 1], self.norm_name, only_content=True)
            self.body_content.append(cur_block_content)
            out_channels = 1 if not self.use_masks else self.num_mask_channels + 1
            self.final_content.append(to_decision(num_of_channels[i + 1], out_channels))

        # --------- D layout --------- #
        self.layout_FA = Layout_FA(self.use_masks, self.prob_FA["layout"], self.save_path)
        for i in range(self.num_blocks_ll, self.num_blocks):
            k = i - self.num_blocks_ll
            in_channels = 1 if k > 0 else num_of_channels[i]
            cur_block_layout = D_block(in_channels, 1, self.norm_name)
            self.body_layout.append(cur_block_layout)
            self.final_layout.append(to_decision(1, 1))
        print("Created Discriminator (%d+%d blocks) with %d parameters" %
              (self.num_blocks_ll, self.num_blocks-self.num_blocks_ll, sum(p.numel() for p in self.parameters())))

    # ------------------------------------------------------------------------
    def content_masked_attention(self, y, mask, for_real, epoch):
        mask  = F.interpolate(mask, size=(y.shape[2], y.shape[3]), mode="nearest")
        y_ans = torch.zeros_like(y).repeat(mask.shape[1], 1, 1, 1)
        if not for_real:
            mask_soft = mask
            if epoch < self.bernoulli_warmup:
                mask_hard = torch.bernoulli(torch.clamp(mask, 0.001, 0.999))
            else:
                mask_hard = F.one_hot(torch.argmax(mask, dim=1), num_classes=mask_soft.shape[1]).permute(0, 3, 1, 2)
            mask = mask_hard - mask_soft.detach() + mask_soft
        for i_ch in range(mask.shape[1]):
            y_ans[i_ch * (y.shape[0]):(i_ch + 1) * (y.shape[0])] = mask[:, i_ch:i_ch + 1, :, :] * y
        return y_ans

    # ------------------------------------------------------------------------
    def debug_lowlevel_feature(self, y, images, for_real, level, epoch):
        if epoch % 500 == 0:
            # feature
            featuremap = y.detach().cpu().numpy()

            # 计算均值
            featuremap = np.mean(featuremap, axis=(1))

            # image
            show_image = images[-level - 1].detach().cpu().numpy().transpose(0, 2, 3, 1)
            show_image = show_image * 0.5 + 0.5

            # 进行显示 对featuremap 进行1 * featuremap.shape[0]的方式进行显示
            plt.figure(figsize=(10, 4))
            for k in range(featuremap.shape[0]):
                plt.subplot(2, featuremap.shape[0], k + 1)
                plt.imshow(featuremap[k])
                plt.axis('off')

                plt.subplot(2, featuremap.shape[0], k + 1 + featuremap.shape[0])
                plt.imshow(show_image[k])
                plt.axis('off')
            plt.tight_layout() 
            if for_real: 
                save_path = 'LL_' + str(level) + "_Epoch_" + str(epoch).zfill(8) + '_true.png'
                plt.suptitle('LowLevel-Feature-TrueImage_Epoch_' + str(epoch))       
            else:
                save_path = 'LL_' + str(level) + "_Epoch_" + str(epoch).zfill(8) + '_fake.png'
                plt.suptitle('LowLevel-Feature-FakeImage_Epoch_' + str(epoch))
            save_path = os.path.join(self.save_path, save_path) if self.save_path else save_path
            plt.savefig(save_path)
            plt.close()
    
    # ------------------------------------------------------------------------
    def debug_layout_feature(self, y_lay, images, for_real, level, epoch):
        # --------- Debug show output_layout featuremap --------- #
        if epoch % 500 == 0:
            if abs(-level-1) > len(images):
                return
            
            # feature
            featuremap = y_lay.detach().cpu().numpy().squeeze()

            # image            
            show_image = images[-level - 1].detach().cpu().numpy().transpose(0, 2, 3, 1)
            show_image = show_image * 0.5 + 0.5

            # 进行显示 对featuremap 进行1 * featuremap.shape[0]的方式进行显示
            if len(featuremap.shape) == 1:
                return

            plt.figure(figsize=(10, 4))
            for j in range(featuremap.shape[0]):
                cur_feature = featuremap[j]
                plt.subplot(2, featuremap.shape[0], j + 1)
                plt.imshow(cur_feature)
                plt.axis('off')

                cur_show = show_image[j]
                plt.subplot(2, featuremap.shape[0], j + 1 + featuremap.shape[0])
                plt.imshow(cur_show)
                plt.axis('off')

                plt.tight_layout()
            plt.suptitle('Layout-Feature-Image_Epoch_' + str(epoch))            
            if for_real: 
                save_path = 'Layout_' + str(level) + "_Epoch_" + str(epoch).zfill(8) + '_true.png'
            else:
                save_path = 'Layout_' + str(level) + "_Epoch_" + str(epoch).zfill(8) + '_fake.png'
            save_path = os.path.join(self.save_path, save_path) if self.save_path else save_path
            plt.savefig(save_path)
            plt.close()

    def discriminate(self, inputs, for_real, epoch):
        images = inputs["images"]
        masks  = inputs["masks"] if self.use_masks else None
        output_ll, output_content, output_layout = list(), list(), list(),
        
        # --------- D low-level --------- #
        y = self.rgb_to_features[0](images[-1])
        for i in range(0, self.num_blocks_ll):
            if i > 0:
                y = torch.cat((y, self.rgb_to_features[i](images[-i - 1])), dim=1)
            y = self.body_ll[i](y)
            
            # Tim.Zhang 只考虑1/8下的结果 提高样本的多样性
            if i > 1 or 1:
                output_ll.append(self.final_ll[i](y))

            # --------- Debug show output_ll featuremap --------- #
            self.debug_lowlevel_feature(y, images, for_real, i, epoch)

        # --------- D content --------- #
        y_con = y
        if self.use_masks:
            y_con = self.content_masked_attention(y, masks, for_real, epoch)
        y_con = torch.mean(y_con, dim=(2, 3), keepdim=True)
        
        if for_real:
            y_con = self.content_FA(y_con, epoch)
        for i in range(self.num_blocks_ll, self.num_blocks):
            k     = i - self.num_blocks_ll
            y_con = self.body_content[k](y_con)

            # Tim.Zhang 考虑1/16 1/32 1/64 1/128下的结果 提高样本的多样性
            output_content.append(self.final_content[k](y_con))

        # --------- D layout --------- #
        y_lay = y
        if for_real:
             y_lay = self.layout_FA(y, masks, epoch)
        for i in range(self.num_blocks_ll, self.num_blocks):
            k     = i - self.num_blocks_ll
            y_lay = self.body_layout[k](y_lay)
            
            # Tim.Zhang 考虑1/16 1/32 和 1/64 1/128下的结果 提高样本的多样性
            output_layout.append(self.final_layout[k](y_lay))

            # --------- Debug show output_layout featuremap --------- #
            self.debug_layout_feature(y_lay, images, for_real, k, epoch)
            
        return {"low-level": output_ll, "content": output_content, "layout": output_layout}


class D_block(nn.Module):
    def __init__(self, in_channel, out_channel, norm_name, is_first=False, only_content=False):
        super(D_block, self).__init__()
        middle_channel = min(in_channel, out_channel)
        ker_size, padd_size = (1, 0) if only_content else (3, 1)
        self.is_first = is_first
        self.activ = nn.LeakyReLU(0.2)
        self.conv1 = sp_norm(nn.Conv2d(in_channel, middle_channel, ker_size, padding=padd_size))
        self.conv2 = sp_norm(nn.Conv2d(middle_channel, out_channel, ker_size, padding=padd_size))
        self.norm1 = get_norm_by_name(norm_name, in_channel)
        self.norm2 = get_norm_by_name(norm_name, middle_channel)
        self.down = nn.AvgPool2d(2) if not only_content else torch.nn.Identity()
        learned_sc = in_channel != out_channel or not only_content
        if learned_sc:
            self.conv_sc = sp_norm(nn.Conv2d(in_channel, out_channel, (1, 1), bias=False))
        else:
            self.conv_sc = torch.nn.Identity()

    def forward(self, x):
        h = x
        if not self.is_first:
            x = self.norm1(x)
            x = self.activ(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activ(x)
        x = self.conv2(x)
        if not x.shape[0] == 0:
            x = self.down(x)
        h = self.conv_sc(h)
        if not x.shape[0] == 0:
            h = self.down(h)
        return x + h
    
