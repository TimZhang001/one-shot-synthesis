import os
import torch
import warnings
import cv2
import albumentations as Augment
import numpy as np
from PIL import Image
from torchvision import transforms as TR
import torchvision.transforms.functional as F
from .recommended_config import get_recommended_config


def prepare_dataloading(opt):
    dataset = Dataset(opt)
    recommended_config = {"image resolution": dataset.image_resolution,
                          "noise_shape": dataset.recommended_config[0],
                          "num_blocks_g":  dataset.recommended_config[1],
                          "num_blocks_d":  dataset.recommended_config[2],
                          "num_blocks_d0": dataset.recommended_config[3],
                          "use_masks": dataset.use_masks,
                          "num_mask_channels": dataset.num_mask_channels}
    if recommended_config["use_masks"] and opt.use_masks:
        print("Using the training regime *with* segmentation masks")
    else:
        opt.use_masks = False
        print("Using the training regime *without* segmentation masks")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = opt.batch_size, shuffle = True, num_workers=8)
    return dataloader, recommended_config


class Dataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        """
        The dataset class. Supports both regimes *with* and *without* segmentation masks.
        """
        self.device = opt.device
        # --- images --- #
        self.root_images = os.path.join(opt.dataroot, opt.dataset_name, "image")
        self.root_masks  = os.path.join(opt.dataroot, opt.dataset_name, "mask")
        self.list_imgs   = self.get_frames_list(self.root_images)
        assert len(self.list_imgs) > 0, "Found no images"
        self.image_resolution, self.recommended_config = get_recommended_config(self.get_im_resolution(opt.max_size))

        # --- masks --- #
        if os.path.isdir(self.root_masks) and opt.use_masks:
            self.list_masks = self.get_frames_list(self.root_masks)
            assert len(self.list_imgs) == len(self.list_masks), \
                "Different number of images and masks %d vs %d" % (len(self.list_imgs), len(self.list_masks))
            for i in range(len(self.list_imgs)):
                assert os.path.splitext(self.list_imgs[i])[0] == os.path.splitext(self.list_masks[i])[0], \
                "Image and its mask must have same names %s - %s" % (self.list_imgs[i], self.list_masks[i])
            self.num_mask_channels = self.get_num_mask_channels()
            self.use_masks         = True
        else:
            self.use_masks = False
            self.num_mask_channels = None

        print("Created a dataset of size =", len(self.list_imgs), "with image resolution", self.image_resolution)

    def get_frames_list(self, path):
        return sorted(os.listdir(path))

    def __len__(self):
        return 100000000  # so first epoch finishes only with break
    
    # 根据image对应的mask，将缺陷区域copy到Object区域
    # mask最多只有3个值，0，1，2 或者 0，1，如果是0，1，2，那么0是背景，1是object，2是Object上的缺陷，如果是0，1，那么0是背景，1是缺陷
    def copy_paste_defect(self, img, mask):
        # 判断mask包含几个值
        assert len(mask.unique()) == 2 or  len(mask.unique()) == 3, "mask must have 2 or 3 unique values"
        
        if len(mask.unique()) == 2:
            # 找到mask==1的区域，得到对应的Rect
            # 进行图像的二值化操作
            mask_bw = np.where(mask >= 1, 255, 0).astype(np.uint8)

            # 统计连通域
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bw, connectivity=8)
            
            # 计算连通域对应的外接矩形
            rects = []
            for i in range(1, num_labels):
                rects.append(cv2.boundingRect(labels == i))
            
            # img和mask
            sub_images = img(rects)
            sub_masks  = mask(rects)

            # 将sub_images中的缺陷区域copy到img中



            pass
        else:
            pass

    
    def get_augmentations(self, target_size):

        # 进行resize
        aug_resize      = Augment.Resize(height=target_size[0], width=target_size[1], p=1)
        
        # 进行水平垂直翻转
        aug_flip        = Augment.Compose([Augment.VerticalFlip(p=0.5), Augment.HorizontalFlip(p=0.5),])
        
        # 90度旋转
        aug_rotate90    = Augment.RandomRotate90(p=0.5)

        # 随机旋转
        #aug_scaleRotate = Augment.ShiftScaleRotate(shift_limit=0.10, scale_limit=0.10, rotate_limit=45, p=0.5)

        # 亮度/对比度拉升
        aug_brtContrast = Augment.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.5)
        
        # 增加GaussNoise噪声
        #aug_noise       = Augment.GaussNoise(p=0.5)

        # RandomGravel
        #aug_gravel      = Augment.RandomGravel(p=0.5)

        # HueSaturationValue
        aug_hueSat      = Augment.HueSaturationValue(p=0.5)

        # RGBShift
        aug_rgbShift    = Augment.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5)

        # MultiplicativeNoise
        # aug_multiNoise  = Augment.MultiplicativeNoise(p=0.5)

        # FancyPCA
        #aug_fancyPCA    = Augment.FancyPCA(p=0.5)

        # ColorJitter
        aug_colorJitter = Augment.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5)

        # Sharpen
        #aug_sharpen     = Augment.Sharpen(p=0.5, alpha=(0.1, 0.25), lightness=(0.75, 1.0))

        # PixelDropout
        # aug_pixelDrop   = Augment.PixelDropout(p=0.5)

        # 组合
        self.augment_fun = Augment.Compose([aug_resize, aug_flip, aug_rotate90, aug_brtContrast,
                                            aug_hueSat, aug_rgbShift,
                                            aug_colorJitter])

    def get_im_resolution(self, max_size):
        """
        Iterate over images to determine image resolution.
        If there are images with different shapes, return the square of average size
        """
        res_list = list()
        for cur_img in self.list_imgs:
            img_pil = Image.open(os.path.join(self.root_images, cur_img)).convert("RGB")
            res_list.append(img_pil.size)
        all_res_equal = len(set(res_list)) <= 1
        if all_res_equal:
            size_1, size_2 = res_list[0]  # all images have same resolution -> using original resolution
        else:
            warnings.warn("Images in the dataset have different resolutions. Resizing them to squares of mean size.")
            size_1 = size_2 = sum([sum(item) for item in res_list]) / (2 * len(res_list))
        size_1, size_2 = self.bound_resolution(size_1, size_2, max_size)
        return size_2, size_1

    def bound_resolution(self, size_1, size_2, max_size):
        """
        Ensure the image shape does not exceed --max_size
        """
        if size_1 > max_size:
            size_1, size_2 = max_size, size_2 / (size_1 / max_size)
        if size_2 > max_size:
            size_1, size_2 = size_1 / (size_2 / max_size), max_size
        return int(size_1), int(size_2)

    def get_num_mask_channels(self):
        """
        Iterate over all masks to determine how many classes are there
        """
        max_index = 0
        for cur_mask in self.list_masks:
            im = TR.functional.to_tensor(Image.open(os.path.join(self.root_masks, cur_mask)))
            if (im.unique() * 256).max() > 30:
                # --- black-white map of one object and background --- #
                max_index = 2 if max_index < 2 else max_index
            else:
                # --- multiple semantic objects --- #
                cur_max   = torch.max(torch.round(im * 256))
                max_index = cur_max + 1 if max_index < cur_max + 1 else max_index
        return int(max_index)

    def create_mask_channels(self, mask):
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
            mask     = torch.nn.functional.one_hot(integers.long(), num_classes=self.num_mask_channels)
            mask     = mask.float()[0].permute(2, 0, 1)
            return mask

    def __getitem__(self, index):
        output      = dict()
        idx         = index % len(self.list_imgs)
        target_size = self.image_resolution

        # --- read image and mask --- #
        img_pil  = Image.open(os.path.join(self.root_images, self.list_imgs[idx])).convert("RGB")
        if self.use_masks:
            mask_pil = Image.open(os.path.join(self.root_masks, self.list_imgs[idx][:-4] + ".png"))
        
        # --- augmentations --- #
        if 1:
            self.get_augmentations(target_size= target_size)
            if self.use_masks:
                # 转化为float32
                #img_pil     = img_pil.astype(np.float32)
                #mask_pil    = mask_pil.astype(np.float32)
                img_augment = self.augment_fun(image = np.array(img_pil), mask = np.array(mask_pil))
                img_pil     = Image.fromarray(img_augment["image"])
                mask_pil    = Image.fromarray(img_augment["mask"])
            else:
                #img_pil     = img_pil.astype(np.float32)
                img_augment = self.augment_fun(image = np.array(img_pil))
                img_pil     = Image.fromarray(img_augment["image"])
            

        # --- image ---#
        #img_pil = Image.open(os.path.join(self.root_images, self.list_imgs[idx])).convert("RGB")
        img     = F.to_tensor(F.resize(img_pil, size=target_size))
        img     = (img - 0.5) * 2
        output["images"] = img

        # --- mask ---#
        if self.use_masks:
            #mask_pil = Image.open(os.path.join(self.root_masks, self.list_imgs[idx][:-4] + ".png"))
            mask     = F.to_tensor(F.resize(mask_pil, size=target_size, interpolation=Image.NEAREST))
            mask     = self.create_mask_channels(mask)  # mask should be N+1 channels
            output["masks"] = mask
            assert img.shape[1:] == mask.shape[1:], "Image and mask must have same dims %s" % (self.list_imgs[idx])
        return output




