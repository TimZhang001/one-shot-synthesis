import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2

from third_party.tim.cutpaste import CutPasteTensor

# 进行特征维度的数据增强，进行不同的样本通道维度的交换，或者针对某个样本进行通道维度的置零
class Content_FA(nn.Module):
    def __init__(self, use_masks, prob_FA_con, num_mask_channels=None):
        super(Content_FA, self).__init__()
        self.prob      = prob_FA_con
        self.ranges    = (0.10, 0.30)
        self.use_masks = use_masks
        if self.use_masks:
            self.num_mask_channels = num_mask_channels

    def mix(self, y):
        """
        Randomly swap channels of different instances
        """
        bs  = y.shape[0]
        ch  = y.shape[1]
        ans = y
        # ---  --- #
        if random.random() < self.prob:
            for i in range(0, bs - 1, 2):
                num_first = int(ch * (torch.rand(1) * (self.ranges[1]-self.ranges[0]) + self.ranges[0]))
                perm      = torch.randperm(ch)
                ch_first  = perm[:num_first]
                
                # 两个样本在通道维度上交换
                ans[i,     ch_first, :, :] = y[i + 1, ch_first, :, :].clone()
                ans[i + 1, ch_first, :, :] = y[i,     ch_first, :, :].clone()

                # debug ----------------- 
                # 对交换前的y[i] y[i+1]进行可视化 对交换后的ans[i] ans[i+1]进行可视化
                if 0:
                    src_feature  = y[i:i+2].detach().cpu().numpy().squeeze()
                    dst_feature  = ans[i:i+2].detach().cpu().numpy().squeeze()
                    zero_feature = np.zeros_like(src_feature)
                    show_feature = np.concatenate([src_feature, zero_feature, dst_feature], axis=0)
                    plt.figure(figsize=(10, 2), dpi=200)
                    plt.imshow(show_feature)
                    plt.axis("off")
                    plt.show()
                    plt.savefig("Content_mix_test_{}.png".format(i))
                    plt.close()
                # debug -----------------
        return ans

    def drop(self, y):
        """
        Randomly zero out channels
        """
        ch  = y.shape[1]
        ans = y
        if random.random() < self.prob:
            num_first  = int(ch * (torch.rand(1) * (self.ranges[1]-self.ranges[0]) + self.ranges[0]))
            num_second = int(ch * (torch.rand(1) * (self.ranges[1]-self.ranges[0]) + self.ranges[0]))
            perm       = torch.randperm(ch)
            ch_second  = perm[num_first:num_first + num_second]
            ans[:, ch_second, :, :] = 0

            # debug ----------------- 
            if 0:
                src_feature  = y.detach().cpu().numpy().squeeze()
                dst_feature  = ans.detach().cpu().numpy().squeeze()
                zero_feature = np.zeros_like(src_feature)
                show_feature = np.concatenate([src_feature, zero_feature, dst_feature], axis=0)
                plt.figure(figsize=(10, 2), dpi=200)
                plt.imshow(show_feature)
                plt.axis("off")
                plt.show()
                plt.savefig("Content_drop_test.png")
                plt.close()
            # debug -----------------
        return ans

    def forward(self, y):
        ans   = y

        # --- Apply only on background if masks are given --- #
        if self.use_masks: 
            y = ans[:ans.shape[0]//self.num_mask_channels]
        
        y = self.mix(y)
        y = self.drop(y)

        # --- Apply only on background if masks are given --- #
        if self.use_masks:
            ans[:ans.shape[0]//self.num_mask_channels] = y
        else:
            ans = y

        return ans

# ---------------------------------------------------------------------------------------------------------------

# 进行特征图的数据增强，
class Layout_FA(nn.Module):
    def __init__(self, use_masks, prob):
        super(Layout_FA, self).__init__()
        self.use_masks = use_masks
        self.prob      = prob
        self.ranges    = (0.10, 0.30)
        self.cutpaste  = CutPasteTensor()

    def forward(self, y, masks):
        if self.use_masks:
            mask_FA = torch.nn.functional.interpolate(masks, size=(y.shape[2], y.shape[3]), mode="nearest")
            ans     = self.func_with_mask(y, mask_FA)
        else:
            ans = self.func_without_mask(y)
            ans = self.func_without_mask_cut_paste(ans)
        return ans

    # 随机生成一个矩形，进行两个样本在特征图上的交换
    def func_without_mask(self, y):
        """
        If a segmentation mask is not provided, copy-paste rectangles in a random way
        """
        bs  = y.shape[0]
        ans = y.clone()
        for i in range(0, bs - 1, 2):
            if random.random() < self.prob:
                x1, x2, y1, y2 = gen_rectangle(ans)
                ans[i,     :, x1:x2, y1:y2] = y[i + 1, :, x1:x2, y1:y2].clone()
                ans[i + 1, :, x1:x2, y1:y2] = y[i,     :, x1:x2, y1:y2].clone()
        return ans
    
    def func_without_mask_cut_paste(self, y):
        bs  = y.shape[0]
        ans = y.clone()

        for i in range(0, bs):
            if random.random() < self.prob:
                ans[i] = self.cutpaste(y[i])
        return ans

    def func_with_mask(self, y, mask):
        """
        If a segmentation mask is provided, ensure that the copied areas never cut semantic boundaries
        """
        ans_y    = y.clone()
        ans_mask = mask.clone()
        ans_y, ans_mask = self.mix_background(ans_y, ans_mask)
        ans_y, ans_mask = self.swap(ans_y, ans_mask)
        ans_y, ans_mask = self.move_objects(ans_y, ans_mask)
        return ans_y

    def mix_background(self, y, mask):
        """
        Copy-paste areas of background onto other background areas
        """
        for i in range(0, y.shape[0]):
            if random.random() < self.prob:
                
                # 生成两个不重叠的矩形框，且不与语义分割图的语义边界重叠
                rect1, rect2 = gen_nooverlap_rectangles(y, mask)
                if rect1[0] is not None:
                    x0_1, x0_2, y0_1, y0_2 = rect1
                    x1_1, x1_2, y1_1, y1_2 = rect2
                    y[i,    :, x0_1:x0_2, y0_1:y0_2] = y[i,    :, x1_1:x1_2, y1_1:y1_2].clone()
                    mask[i, :, x0_1:x0_2, y0_1:y0_2] = mask[i, :, x1_1:x1_2, y1_1:y1_2].clone()

                    # debug ----------------- 
                    if 0:
                        src_feature  = y[i].detach().cpu().numpy().squeeze()
                        src_feature  = np.mean(src_feature, axis=0)
                        # 绘制 rect1 rect2
                        cv2.rectangle(src_feature, (y0_1, x0_1), (y0_2, x0_2), (0, 255, 0), 1)
                        cv2.rectangle(src_feature, (y1_1, x1_1), (y1_2, x1_2), (0, 128, 0), 1)
                        mask_feature = mask[i].detach().cpu().numpy().squeeze()
                        mask_feature = np.transpose(mask_feature, (1, 2, 0))
                        plt.figure(figsize=(6, 2), dpi=200)
                        plt.subplot(1, 2, 1)
                        plt.imshow(src_feature)
                        plt.axis("off")
                        plt.subplot(1, 2, 2)
                        plt.imshow(mask_feature)
                        plt.axis("off")
                        plt.show()
                        plt.savefig("Layout_mix_background_{}.png".format(i))
                        plt.close()
                    # debug -----------------


        return y, mask

    def swap(self, y, mask_):
        """
        Copy-paste background and objects into other areas, without cutting semantic boundaries
        """
        ans  = y.clone()
        mask = mask_.clone()
        for i in range(0, y.shape[0] - 1, 2):
            if random.random() < self.prob:
                for jj in range(5):
                    x1, x2, y1, y2 = gen_rectangle(y)
                    rect           = x1, x2, y1, y2
                    if any_object_touched(rect, mask[i:i + 1]) or any_object_touched(rect, mask[i + 1:i + 2]):
                        continue
                    else:
                        ans[i,     :, x1:x2, y1:y2]  = y[i + 1, :, x1:x2, y1:y2].clone()
                        ans[i + 1, :, x1:x2, y1:y2]  = y[i,     :, x1:x2, y1:y2].clone()
                        mem                          = mask_[i,     :, x1:x2, y1:y2].clone()
                        mask[i,     :, x1:x2, y1:y2] = mask_[i + 1, :, x1:x2, y1:y2].clone()
                        mask[i + 1, :, x1:x2, y1:y2] = mem

                        # debug ----------------- 
                        if 0:
                            src_feature  = y[i].detach().cpu().numpy().squeeze()
                            src_feature  = np.mean(src_feature, axis=0)
                            cv2.rectangle(src_feature, (y1, x1), (y2, x2), (0, 255, 0), 1)
                            dst_feature  = y[i+1].detach().cpu().numpy().squeeze()
                            dst_feature  = np.mean(dst_feature, axis=0)
                            cv2.rectangle(dst_feature, (y1, x1), (y2, x2), (0, 128, 0), 1)

                            mask_feature1 = mask[i].detach().cpu().numpy().squeeze()
                            mask_feature1 = np.transpose(mask_feature1, (1, 2, 0))

                            mask_feature2 = mask[i+1].detach().cpu().numpy().squeeze()
                            mask_feature2 = np.transpose(mask_feature2, (1, 2, 0))
                            
                            plt.figure(figsize=(10, 2), dpi=200)
                            plt.subplot(1, 4, 1)
                            plt.imshow(src_feature)
                            plt.axis("off")
                            plt.subplot(1, 4, 2)
                            plt.imshow(dst_feature)
                            plt.axis("off")
                            plt.subplot(1, 4, 3)
                            plt.imshow(mask_feature1)
                            plt.axis("off")
                            plt.subplot(1, 4, 4)
                            plt.imshow(mask_feature2)
                            plt.axis("off")
                            plt.show()
                            plt.savefig("Layout_swap_{}.png".format(i))
                            plt.close()
                        # debug -----------------
                        break
            if random.random() < self.prob:
                which_object = torch.randint(mask.shape[1] - 1, size=()) + 1
                old_area     = torch.argmax(mask[i], dim=0, keepdim=False) == which_object
                if not area_cut_any_object(old_area, mask[i + 1]):
                    ans[i+1]  = ans[i].clone() * (old_area * 1.0) + ans[i+1].clone() * (1 - old_area * 1.0)
                    mask[i+1] = mask[i]        * (old_area * 1.0) + mask[i+1]        * (1 - old_area * 1.0)
        return ans, mask

    def move_objects(self, y, mask):
        """
        Move, dupplicate, or remove semantic objects
        """
        for i in range(0, y.shape[0]):
            num_changed_objects = torch.randint(mask.shape[1] - 1, size=()) + 1
            seq_classes         = torch.randperm(mask.shape[1] - 1)[:num_changed_objects]
            for cur_class in seq_classes:
                old_area = torch.argmax(mask[i], dim=0, keepdim=False) == cur_class + 1  # +1 to avoid background
                new_area = generate_new_area(old_area, mask[i])
                if new_area[0] is None:
                    continue
                if random.random() < self.prob:
                    y[i], mask[i] = dupplicate_object(y[i], mask[i], old_area, new_area)
                if random.random() < self.prob:
                    y[i], mask[i] = remove_object(y[i], mask[i], old_area, new_area)
            return y, mask


# 生成矩形框
def gen_rectangle(ans, w=-1, h=-1):
    x_c, y_c = random.random(), random.random()
    x_s, y_s = random.random()*0.2+0.1, random.random()*0.2+0.1
    x_l, x_r = x_c-x_s/2, x_c+x_s/2
    y_l, y_r = y_c-y_s/2, y_c+y_s/2
    x1,  x2  = int(x_l*ans.shape[2]), int(x_r*ans.shape[2])
    y1,  y2  = int(y_l*ans.shape[3]), int(y_r*ans.shape[3])
    if w < 0 or h < 0:
        pass
    else:
        x2, y2 = x1 + w, y1 + h
    x1, x2, y1, y2 = trim_rectangle(x1, x2, y1, y2, ans.shape)
    return x1, x2, y1, y2

# 保证生成的矩形框不会超出图像的边界
def trim_rectangle(x1, x2, y1, y2, sh):
    if x1 < 0:
        x2 += (0 - x1)
        x1 += (0 - x1)
    if x2 >= sh[2]:
        x1 -= (x2 - sh[2] + 1)
        x2 -= (x2 - sh[2] + 1)
    if y1 < 0:
        y2 += (0 - y1)
        y1 += (0 - y1)
    if y2 >= sh[3]:
        y1 -= (y2 - sh[3] + 1)
        y2 -= (y2 - sh[3] + 1)
    return x1, x2, y1, y2


def gen_nooverlap_rectangles(ans, mask):
    x0_1, x0_2, y0_1, y0_2 = gen_rectangle(ans)
    for i in range(5):
        x1_1, x1_2, y1_1, y1_2 = gen_rectangle(ans, w=x0_2-x0_1, h=y0_2-y0_1)
        
        # 保证两个矩形框不会重叠
        if not (x0_1 < x1_2 and x0_2 > x1_1 and y0_1 < y1_2 and y0_2 > y1_1):
            rect1, rect2 = [x0_1, x0_2, y0_1, y0_2], [x1_1, x1_2, y1_1, y1_2]
            
            # 保证两个矩形框不会与语义分割图的语义边界重叠
            if not any_object_touched(rect1, mask[i:i + 1]) and not any_object_touched(rect2, mask[i:i + 1]):
                return [x0_1, x0_2, y0_1, y0_2], [x1_1, x1_2, y1_1, y1_2]
    return [None, None, None, None], [None, None, None, None]  # if not found a good pair


def any_object_touched(rect, mask_):
    epsilon            = 0.01
    x1, x2, y1, y2     = rect
    mask               = torch.zeros_like(mask_)
    mask[:, 0, :, :]   = mask_[:, 0, :, :]
    mask[:, 1:2, :, :] = torch.sum(torch.abs(mask_[:, 1:, :, :]), dim=1, keepdim=True)
    sum                = torch.sum(mask[:, 1, x1:x2, y1:y2])
    if sum > epsilon:
        return True
    return False


def area_cut_any_object(area, mask_):
    epsilon = 0.01
    mask = torch.zeros_like(mask_)
    mask[0, :, :] = mask_[0, :, :]
    mask[1:2, :, :] = torch.sum(torch.abs(mask_[1:, :, :]), dim=0, keepdim=True)
    sum = torch.sum(area * mask[1, :, :])
    if sum > epsilon:
        return True
    return False


def generate_new_area(old_area, mask):
    epsilon  = 0.01
    arg_mask = torch.argmax(mask, dim=0)
    if torch.sum(old_area) == 0:
        return None, None, None, None, None, None
    idx_x1 = torch.nonzero(old_area * 1.0)[:, 0].min()
    idx_x2 = torch.nonzero(old_area * 1.0)[:, 0].max()
    idx_y1 = torch.nonzero(old_area * 1.0)[:, 1].min()
    idx_y2 = torch.nonzero(old_area * 1.0)[:, 1].max()
    for i in range(5):
        new_x1 = torch.randint(0, mask.shape[1] - (idx_x2 - idx_x1), size=())
        new_y1 = torch.randint(0, mask.shape[2] - (idx_y2 - idx_y1), size=())
        x_diff = new_x1 - idx_x1
        y_diff = new_y1 - idx_y1
        provisional_area = torch.zeros_like(old_area)
        provisional_area[idx_x1+x_diff:idx_x2+x_diff+1, idx_y1+y_diff:idx_y2+y_diff+1] \
            = old_area[idx_x1:idx_x2 + 1, idx_y1:idx_y2 + 1]
        check_sum = torch.sum((provisional_area * 1.0) * arg_mask)
        if check_sum < epsilon:
            return x_diff, y_diff, idx_x1, idx_x2, idx_y1, idx_y2
    return None, None, None, None, None, None


def dupplicate_object(y, mask, old_area, new_area):
    x_diff, y_diff, idx_x1, idx_x2, idx_y1, idx_y2 = new_area

    y[:, idx_x1 + x_diff:idx_x2 + x_diff + 1, idx_y1 + y_diff:idx_y2 + y_diff + 1] = \
        y[:, idx_x1 + x_diff:idx_x2 + x_diff + 1, idx_y1 + y_diff:idx_y2 + y_diff + 1] \
        * (1.0 - old_area[idx_x1:idx_x2 + 1, idx_y1:idx_y2 + 1] * (1.0)) \
        + y[:, idx_x1:idx_x2 + 1, idx_y1:idx_y2 + 1] \
        * (old_area[idx_x1:idx_x2 + 1,idx_y1:idx_y2 + 1] * 1.0)

    mask[:, idx_x1 + x_diff:idx_x2 + x_diff + 1, idx_y1 + y_diff:idx_y2 + y_diff + 1] = \
        mask[:, idx_x1 + x_diff:idx_x2 + x_diff + 1, idx_y1 + y_diff:idx_y2 + y_diff + 1] \
        * (1.0 - old_area[idx_x1:idx_x2 + 1, idx_y1:idx_y2 + 1] * (1.0)) \
        + mask[:, idx_x1:idx_x2 + 1, idx_y1:idx_y2 + 1] \
        * (old_area[idx_x1:idx_x2 + 1, idx_y1:idx_y2 + 1] * 1.0)
    return y, mask


def remove_object(y, mask, old_area, new_area):
    x_diff, y_diff, idx_x1, idx_x2, idx_y1, idx_y2 = new_area

    y[:, idx_x1:idx_x2 + 1, idx_y1:idx_y2 + 1] = \
        y[:, idx_x1 + x_diff:idx_x2 + x_diff + 1, idx_y1 + y_diff:idx_y2 + y_diff + 1]

    mask[:, idx_x1:idx_x2 + 1, idx_y1:idx_y2 + 1] \
        = mask[:, idx_x1 + x_diff:idx_x2 + x_diff + 1, idx_y1 + y_diff:idx_y2 + y_diff + 1]
    return y, mask
