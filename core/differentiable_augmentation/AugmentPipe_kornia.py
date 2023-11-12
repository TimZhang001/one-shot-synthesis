import torch
import kornia
import random
from torchvision import transforms as TR
import torch.nn.functional as F


class AugmentPipe_kornia(torch.nn.Module):
    def __init__(self, prob, use_masks):
        super().__init__()
        self.prob = prob
        self.use_masks = use_masks

    def forward(self, batch):
        x = batch["images"]
        if self.use_masks:
            mask = batch["masks"]
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
                    
                        
        # 1. 先放大2倍，然后再Crop到原来的尺寸 Tim.Zhang add 2023.11.12
        if random.random() < self.prob/2 and 0:
            tr = kornia.augmentation.RandomCrop(size=(sh[2], sh[3]), same_on_batch=True)
            for i in range(sh[0]):
                x[i] = torch.nn.functional.interpolate(x[i], size=(2*sh[2], 2*sh[3]), mode="bilinear")
                x[i] = tr(x[i])

        if random.random() < self.prob:
            
            # 2. Crop到一个小一点的尺寸(0.75, 1)，然后再resize回来
            if random.random() < 0.5:
                r = random.random() * 0.25 + 0.75
                tr = kornia.augmentation.RandomCrop(size=(int(sh[2]*r), int(sh[3]*r)), same_on_batch=True)
                for i in range(sh[0]):
                    x[i] = tr(x[i])
                    x[i] = torch.nn.functional.interpolate(x[i], size=(sh[2], sh[3]), mode="bilinear")
            # 2. 旋转8度，然后再Crop到一个小一点的尺寸(0.8)，然后再resize回来
            else:
                tr = kornia.augmentation.RandomRotation(degrees=8, same_on_batch=True)
                for i in range(sh[0]):
                    x[i] = tr(x[i])
                tr = kornia.augmentation.CenterCrop(size=(sh[2]*0.80, sh[3]*0.80))
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

        # 5. 水平方向随机裁剪部分复制到另一边
        if random.random() < self.prob:
            for i in range(sh[0]):
                # Tim.Zhang add 2023.11.12
                # x[i] = translate_v_fake(x[i], fraction=(0.05, 0.3))
                x[i] = translate_v_fake(x[i], fraction=(0.05, 0.10))
        
        # 6. 垂直方向随机裁剪部分复制到另一边
        if random.random() < self.prob:
            for i in range(sh[0]):
                # Tim.Zhang add 2023.11.12
                # x[i] = translate_h_fake(x[i], fraction=(0.05, 0.3))
                x[i] = translate_h_fake(x[i], fraction=(0.05, 0.10))

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
            cur[j, :, :, :] = F.interpolate(inp[j][i, :, :, :].unsqueeze(0), size=(sh[2], sh[3]),
                                                              mode="bilinear")
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


# main
if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt
    import os

    # 对AugmentPipe_kornia的增强效果进行测试
    AugmentPipe_kornia = AugmentPipe_kornia(0.3, 1)

    input_image = cv2.imread('/home/zhangss/PHDPaper/06_OneShotSynthesis1/datasets/mvtec/image/005.png') 
    input_image = cv2.resize(input_image, (320, 320))


    for i in range(1000):
        batch = dict()
        batch["images"] = list()
        scale_val       = [64, 32, 16, 8, 4, 2, 1]
        for k in range(7):
            cur_image = cv2.resize(input_image, (320 // scale_val[k], 320 // scale_val[k]))
            cur_image = cv2.cvtColor(cur_image, cv2.COLOR_BGR2RGB)
            cur_image = torch.from_numpy(cur_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            cur_image = cur_image.repeat(1, 1, 1, 1)
            batch["images"].append(cur_image)

        # 7张图，按照2*7的排列方式显示
        fig = plt.figure(figsize=(10, 3), dpi=200)
   
        for j in range(7):
            sub_image = batch["images"][j].squeeze(0).permute(1, 2, 0).numpy()
            fig.add_subplot(2, 7, j+1)
            plt.imshow(sub_image)
            plt.axis('off')
        
        output_image = AugmentPipe_kornia(batch)["images"]
        
        for j in range(7):
            sub_image = output_image[j].squeeze(0).permute(1, 2, 0).numpy()
            fig.add_subplot(2, 7, j+8)
            plt.imshow(sub_image)
            plt.axis('off')

        base_path = '/home/zhangss/PHDPaper/06_OneShotSynthesis1/output/aug_image/'
        os.makedirs(base_path, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(base_path, f'output_image_{i}.png'))
        plt.close()