import torch
import kornia
import random
from torchvision import transforms as TR
import torch.nn.functional as F
import albumentations as Aug
from PIL import Image
import torchvision.transforms.functional as TransF

class AugmentPipe_kornia_New(torch.nn.Module):
    def __init__(self, prob, use_masks):
        super().__init__()
        self.prob      = prob
        self.use_masks = use_masks

    def forward(self, batch):
        x = batch["images"]
        if self.use_masks:
            mask    = batch["masks"]
            mask_ch = mask.shape[1]
            
        ref = x
        if self.use_masks:
            new_mask = mask.clone()
        sh = x[-1].shape
        x = combine_fakes(x)

        if self.use_masks:
            for i in range(len(x)):
                for ch in range(mask_ch, 0, -3):
                    if ch > 2:
                        x[i] = torch.cat((x[i], mask[i].unsqueeze(0)[:, ch-3:ch, :, :]), dim=(0))
                    else:
                        x[i] = torch.cat((x[i], mask[i].unsqueeze(0).repeat(1, 4-ch, 1, 1)[:, :3, :, :]), dim=(0))
                    

        # 1. Crop到一个小一点的尺寸(0.85, 1)，然后再resize回来                
        if random.random() < self.prob:       
            r = random.random() * 0.15 + 0.85
            tr = kornia.augmentation.RandomCrop(size=(int(sh[2]*r), int(sh[3]*r)), same_on_batch=True)
            for i in range(sh[0]):
                x[i] = tr(x[i])
                x[i] = torch.nn.functional.interpolate(x[i], size=(sh[2], sh[3]), mode="bilinear")

        # 2. 旋转-45，,45度，然后再Crop到一个小一点的尺寸(0.95)，然后再resize回来
        if random.random() < self.prob:
            tr = kornia.augmentation.RandomRotation(degrees=45, same_on_batch=True)
            for i in range(sh[0]):
                x[i] = tr(x[i])
            tr = kornia.augmentation.CenterCrop(size=(sh[2]*0.95, sh[3]*0.90))
            for i in range(sh[0]):
                x[i] = tr(x[i])
                x[i] = torch.nn.functional.interpolate(x[i], size=(sh[2], sh[3]), mode="bilinear")

        # 3. 水平翻转
        if random.random() < self.prob:
            tr = kornia.augmentation.RandomHorizontalFlip(p=1.0, same_on_batch=True)
            for i in range(sh[0]):
                x[i] = tr(x[i])

        # 4. 垂直翻转
        if random.random() < self.prob:
            tr = kornia.augmentation.RandomVerticalFlip(p=1.0, same_on_batch=True)
            for i in range(sh[0]):
                x[i] = tr(x[i])

        # 5. 长宽比例变换
        if random.random() < self.prob:
            tr = kornia.augmentation.RandomResizedCrop(size=(sh[2], sh[3]), scale=(0.8, 1.0), ratio=(0.7, 1.3), same_on_batch=True)
            for i in range(sh[0]):
                x[i] = tr(x[i])
        
        # 6. 色彩度变换
        if random.random() < self.prob and 0:
            tr = kornia.augmentation.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.025, same_on_batch=True)
            for i in range(sh[0]):
                x[i] = tr(x[i])
        
        # 7. 透视变换
        if random.random() < self.prob:
            tr = kornia.augmentation.RandomPerspective(p=1.0, distortion_scale= 0.1, same_on_batch=True)
            for i in range(sh[0]):
                x[i] = tr(x[i])

        if self.use_masks:
            for i in range(len(x)):
                for ch in reversed(range(mask_ch, 0, -3)):
                    if ch > 2:
                        new_mask[i, ch-3:ch] = x[i][-1-(ch-1)//3]
                    else:
                        new_mask[i, :ch] = x[i][-1][:ch]
                x[i] = x[i][:-1-(ch-1)//3]
        x = detach_fakes(x, ref)

        batch["images"] = x
        if self.use_masks:
            batch["masks"] = new_mask
        return batch


def combine_fakes(inp):
    sh = inp[-1].shape
    ans = list()
    for i in range(sh[0]):
        cur = torch.zeros_like(inp[-1][0, :, :, :]).repeat(len(inp), 1, 1, 1)
        for j in range(len(inp)):
            cur[j, :, :, :] = F.interpolate(inp[j][i, :, :, :].unsqueeze(0), size=(sh[2], sh[3]), mode="bilinear")
        ans.append(cur)
    return ans


def detach_fakes(inp, ref):
    ans = list()
    sh = ref[-1].shape
    for i in range(len(ref)):
        cur = torch.zeros_like(ref[i])
        for j in range(sh[0]):
            cur[j, :, :, :] = F.interpolate(inp[j][i, :, :, :].unsqueeze(0),
                                                              size=(ref[i].shape[2], ref[i].shape[3]),
                                                              mode="bilinear")
        ans.append(cur)
    return ans


class myRandomResizedCrop(TR.RandomResizedCrop):
    def __init__(self, size=256, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), ):
        super(myRandomResizedCrop, self).__init__(size, scale, ratio)

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return TR.functional.resized_crop(img, i, j, h, w, (img.size[1], img.size[0]), self.interpolation)


def translate_v_fake(x, fraction):
    margin    = torch.rand(1) * (fraction[1] - fraction[0]) + fraction[0]
    direct_up = (torch.rand(1) < 0.5)  # up or down
    height, width = x.shape[2], x.shape[3]
    left, right = 0, width
    if direct_up:
        top, bottom = 0, int(height * margin)
    else:
        top, bottom = height - int(height * margin), height
    im_to_paste = torch.flip(x[:, :, top:bottom, left:right], (2,))
    if not direct_up:
        x[:, :, 0:height - int(height * margin), :] = x[:, :, int(height * margin):height, :].clone()
        x[:, :, height - int(height * margin):, :] = im_to_paste
    else:
        x[:, :, int(height * margin):height, :] = x[:, :, 0:height - int(height * margin), :].clone()
        x[:, :, :int(height * margin), :] = im_to_paste
    return x


def translate_h_fake(x, fraction):
    margin = torch.rand(1) * (fraction[1] - fraction[0]) + fraction[0]
    direct_left = (torch.rand(1) < 0.5)  # up or down
    height, width = x.shape[2], x.shape[3]
    top, bottom = 0, height
    if direct_left:
        left, right = 0, int(width * margin)
    else:
        left, right = width - int(width * margin), width
    im_to_paste = torch.flip(x[:, :, top:bottom, left:right], (3,))
    if not direct_left:
        x[:, :, :, 0:width - int(width * margin)] = x[:, :, :, int(width * margin):width].clone()
        x[:, :, :, width - int(width * margin):] = im_to_paste
    else:
        x[:, :, :, int(width * margin):width] = x[:, :, :, 0:width - int(width * margin)].clone()
        x[:, :, :, :int(width * margin)] = im_to_paste
    return x


def create_mask_channels(mask, num_mask_channels):
    """
    Convert a mask to one-hot representation
    """
    if (mask.unique() * 256).max() > 30:
        # --- only object and background--- #
        mask = torch.cat((1 - mask, mask), dim=0)
        return mask
    else:
        # --- multiple semantic objects --- #
        integers = torch.round(mask * 256)
        mask     = torch.nn.functional.one_hot(integers.long(), num_classes=num_mask_channels)
        mask     = mask.float()[0].permute(2, 0, 1)
        return mask

# main
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    # 对AugmentPipe_kornia的增强效果进行测试
    AugmentPipe_kornia = AugmentPipe_kornia_New(1, 1)

    img_pil = Image.open('/home/zhangss/PHDPaper/06_OneShotSynthesis1/datasets/mvtec/image/005.png').convert("RGB")
    img     = TransF.to_tensor(TransF.resize(img_pil, size=(320, 320)))
    img     = (img - 0.5) * 2

    mask_pil = Image.open('/home/zhangss/PHDPaper/06_OneShotSynthesis1/datasets/mvtec/mask/005.png')
    mask     = TransF.to_tensor(TransF.resize(mask_pil, size=(320, 320), interpolation=Image.NEAREST))
    mask     = create_mask_channels(mask, 3)  # mask should be N+1 channels

    for i in range(1000):
        batch           = dict()
        batch["images"] = list()
        
        mask_image  = mask.repeat(1, 1, 1, 1)
        input_image = img.repeat(1, 1, 1, 1)
        cur_image   = input_image.clone()
        
        batch["images"].append(cur_image)
        for k in range(6):
            cur_image = F.interpolate(cur_image, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False)
            batch["images"].append(cur_image)
        
        batch["images"] = list(reversed(batch["images"]))
        batch["masks"]  = mask_image.clone()

        # 7张图，按照3*7的排列方式显示
        fig = plt.figure(figsize=(10, 3), dpi=200)
        for j in range(7):
            sub_image = batch["images"][j].squeeze(0).permute(1, 2, 0).numpy()
            sub_image = (sub_image / 2.0) + 0.5
            fig.add_subplot(3, 7, j+1)
            plt.imshow(sub_image)
            plt.axis('off')
        
        batch_out = AugmentPipe_kornia(batch)
     
        for j in range(7):
            sub_image = batch_out["images"][j].squeeze(0).permute(1, 2, 0).numpy()
            sub_image = (sub_image / 2.0) + 0.5
            fig.add_subplot(3, 7, j+8)
            plt.imshow(sub_image)
            plt.axis('off')
        
        sub_out_mask = batch_out["masks"].squeeze(0).permute(1, 2, 0).numpy()
        fig.add_subplot(3, 7, 15)
        plt.imshow(sub_out_mask, vmin=0.0, vmax=2.0)
        plt.axis('off')

        sub_in_mask  = mask_image.squeeze(0).permute(1, 2, 0).numpy()
        fig.add_subplot(3, 7, 16)
        plt.imshow(sub_in_mask, vmin=0.0, vmax=2.0)
        plt.axis('off')

        base_path = '/home/zhangss/PHDPaper/06_OneShotSynthesis1/output/aug_image/'
        os.makedirs(base_path, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(base_path, f'output_image_{i}.png'))
        plt.close()