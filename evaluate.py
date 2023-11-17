"""
This file computes metrics for a chosen checkpoint. By default, it computes SIFID (at lowest InceptionV3 scale),
LPIPS diversity, LPIPS distance to training data, mIoU (in case segmentation masks are used).
The results are saved at /${checkpoints_dir}/${exp_name}/metrics/
For SIFID, LPIPS_to_train, mIoU, and segm accuracy, the metrics are computed per each image.
LPIPS, mIoU and segm_accuracy are also computed for the whole dataset.
"""


import os
import argparse
import numpy as np
import pickle
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from metrics import SIFID, LPIPS, LPIPS_to_train, mIoU
import matplotlib.pyplot as plt
import seaborn as sns

def parser_param():

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints_dir',  type=str,  default='checkpoints')
    parser.add_argument('--exp_name',         type=str,  default='mvtec_grid_no_mask_base')
    parser.add_argument('--epoch',            type=str,  default='00100000')
    parser.add_argument('--sifid_all_layers', type=bool, default=False)
    args = parser.parse_args()

    return args

def convert_sifid_dict(names_fake_image, sifid):
    ans = dict()
    if sifid is not None:
        for i, item in enumerate(names_fake_image):
            ans[item] = sifid[i]
    return ans

def get_image_names(args):
    # --- Read options file from checkpoint --- #
    file_name    = os.path.join(args.checkpoints_dir, args.exp_name, "opt.pkl")
    new_opt      = pickle.load(open(file_name, 'rb'))
    use_masks    = getattr(new_opt, "use_masks")
    dataroot     = getattr(new_opt, "dataroot")
    dataset_name = getattr(new_opt, "dataset_name")
    path_real_images = os.path.join(dataroot, dataset_name, "image")
    if use_masks:
        path_real_masks = os.path.join(dataroot, dataset_name, "mask")
    else:
        path_real_masks = None

    # --- Prepare files and images to compute metrics --- #
    names_real_image = sorted(os.listdir(path_real_images))
    if use_masks:
        names_real_masks = sorted(os.listdir(path_real_masks))
    else:
        names_real_masks = None

    names_fake       = sorted(os.listdir(os.path.join(exp_folder)))
    names_fake_image = [item for item in names_fake if "mask" not in item]
    if use_masks:
        names_fake_masks = [item[:-4]+"_mask"+item[-4:] for item in names_fake_image]
    else:
        names_fake_masks = None

    list_real_image, list_fake_image = list(), list()
    for i in range(len(names_fake_image)):
        im               = (Image.open(os.path.join(exp_folder, names_fake_image[i])).convert("RGB"))
        list_fake_image += [im]

    im_res = (ToTensor()(list_fake_image[0]).shape[2], ToTensor()(list_fake_image[0]).shape[1])
    for i in range(len(names_real_image)):
        im               = (Image.open(os.path.join(path_real_images, names_real_image[i])).convert("RGB"))
        list_real_image += [im.resize(im_res, Image.BILINEAR)]

    return use_masks, names_real_image, names_real_masks, path_real_images, \
        names_fake_image, names_fake_masks, path_real_masks, list_real_image, list_fake_image, im_res

def write_sfid_to_file(save_fld_file, names_fake_image, sifid1, sifid2, sifid3, sifid4, args):
    
    for i in range(1, 4):
        if i == 1:
            tempfid = sifid1
        elif i == 2:
            tempfid = sifid2
        elif i == 3:
            tempfid = sifid3
        elif i == 4:
            tempfid = sifid4

        file_name = os.path.join(save_fld_file, str(args.epoch))+"SIFID"+str(i)+".csv"
      
        # write to npy file and csv file
        if tempfid is not None:
            tempfid        = convert_sifid_dict(names_fake_image, tempfid)
            tempfid_values = np.array(list(tempfid.values()))
            tempfid_mean   = np.mean(tempfid_values)
            tempfid_std    = np.std(tempfid_values)
            tempfid_min    = np.min(tempfid_values)
            tempfid_max    = np.max(tempfid_values)
            tempfid_median = np.median(tempfid_values)
            
            #np.save(os.path.join(save_fld, str(args.epoch))+"SIFID1", sifid1)  
            with open(file_name, "w") as f:
                f.write("ave values, "    + format(tempfid_mean, ".6f") + "\n")
                f.write("std values, "    + format(tempfid_std, ".6f") + "\n")
                f.write("min values, "    + format(tempfid_min, ".6f") + "\n")
                f.write("max values, "    + format(tempfid_max, ".6f") + "\n")
                f.write("median values, " + format(tempfid_median, ".6f") + "\n")

                for k, v in tempfid.items():
                    f.write(str(k) + "," + format(v, ".6f") + "\n")
            print("--- Saved sifid metrics at %s ---" % (file_name))

            plot_hist_fig(tempfid_values, save_fld_file, hist_num = 128, title_name=str(args.epoch)+"SIFID"+str(i))
    
def write_lpips_to_file(save_fld_file, lpips, dist_to_tr, dist_to_tr_byimage, args):
    
    # write to npy file and csv file
    # np.save(os.path.join(save_fld_file, str(args.epoch))+"lpips",              lpips)
    # np.save(os.path.join(save_fld_file, str(args.epoch))+"dist_to_tr",         dist_to_tr)
    # np.save(os.path.join(save_fld_file, str(args.epoch))+"dist_to_tr_byimage", dist_to_tr_byimage)

    # write to csv file
    file_name = os.path.join(save_fld_file, str(args.epoch))+"lpips.csv"   
    with open(file_name, "w") as f:
        # lpips values
        f.write("fake ave values, "    + format(lpips.item(), ".6f") + "\n")
        
        # dist_to_tr values
        f.write("fake to true ave values, " + format(dist_to_tr.item(), ".6f") + "\n")
        
        # dist_to_tr_byimage values
        tempfid_values = np.array(list(dist_to_tr_byimage.values()))
        tempfid_mean   = np.mean(tempfid_values)
        tempfid_std    = np.std(tempfid_values)
        tempfid_min    = np.min(tempfid_values)
        tempfid_max    = np.max(tempfid_values)
        tempfid_median = np.median(tempfid_values)
        f.write("ave values, "    + format(tempfid_mean, ".6f") + "\n")
        f.write("std values, "    + format(tempfid_std, ".6f") + "\n")
        f.write("min values, "    + format(tempfid_min, ".6f") + "\n")
        f.write("max values, "    + format(tempfid_max, ".6f") + "\n")
        f.write("median values, " + format(tempfid_median, ".6f") + "\n")
        for k, v in dist_to_tr_byimage.items():
            f.write(str(k) + "," + format(v, ".6f") + "\n")

        plot_hist_fig(tempfid_values, save_fld_file, hist_num = 128, title_name=str(args.epoch)+"_lpips_dist_to_true")

def plot_hist_fig(values_list, save_path, hist_num = 128, title_name=""):

    # plot histogram
    save_name = os.path.join(save_path, title_name+".png")
    fig, ax   = plt.subplots(figsize=(8, 6))
    sns.distplot(values_list, bins=hist_num, kde=False, ax=ax)
    ax.set_title(title_name)
    plt.tight_layout()
    plt.show()
    fig.savefig(save_name)
    plt.close(fig)


if __name__ == "__main__":
    args = parser_param()
    print("--- Computing metrics for job %s at epoch %s ---" %(args.exp_name, args.epoch))

    exp_folder = os.path.join(args.checkpoints_dir, args.exp_name, "evaluation", args.epoch)
    if not os.path.isdir(exp_folder):
        raise ValueError("Generated images not found. Run the test script first. (%s)" % (exp_folder))

    # --- Get image names --- #
    use_masks, names_real_image, names_real_masks, path_real_images, \
        names_fake_image, names_fake_masks, path_real_masks, list_real_image, list_fake_image, im_res = get_image_names(args)

    # --- Compute the metrics --- #
    with torch.no_grad():
        sifid1, sifid2, sifid3, sifid4 = SIFID(list_real_image, list_fake_image, args.sifid_all_layers)
        lpips                          = LPIPS(list_fake_image)
        dist_to_tr, dist_to_tr_byimage = LPIPS_to_train(list_real_image, list_fake_image, names_fake_image)
    if use_masks and 0:
        miou_tensor, miou_byimage, acc_byimage = mIoU(path_real_images, names_real_image, path_real_masks, names_real_masks,
                                                      exp_folder, names_fake_image, names_fake_masks, im_res)
    else:
        miou_tensor, miou_byimage, acc_byimage = None, None, None

    # --- Save the metrics under .${exp_name}/metrics --- #
    save_fld = os.path.join(args.checkpoints_dir, args.exp_name, "metrics")
    os.makedirs(save_fld, exist_ok=True)

    # ----------------------------------------------------------------------------
    write_sfid_to_file(save_fld, names_fake_image, sifid1, sifid2, sifid3, sifid4, args)

    # ----------------------------------------------------------------------------
    write_lpips_to_file(save_fld, lpips.cpu(), dist_to_tr.cpu(), dist_to_tr_byimage, args)

    # ----------------------------------------------------------------------------
    if use_masks and 0:
        # format for segm_miou_accuracy is 1) Accuracy (val->tr) 2) mIoU (val->tr) 4) Accuracy (tr->val) 5) mIoU (tr->val)
        np.save(os.path.join(save_fld, str(args.epoch))+"segm_miou_accuracy",    miou_tensor)
        np.save(os.path.join(save_fld, str(args.epoch))+"segm_accuracy_byimage", acc_byimage)
        np.save(os.path.join(save_fld, str(args.epoch))+"segm_miou_byimage",     miou_byimage)
    
    print("--- Saved metrics at %s ---" % (save_fld))


