
﻿# Overview

This repository implements OSMIS, an unconditional GAN model for one-shot synthesis of images and segmentation masks ([Link to WACV 2023 Paper](https://arxiv.org/abs/2209.07547)). The code also contains the implementation of the one-shot image synthesis baseline [SIV-GAN](https://arxiv.org/abs/2103.13389) (also was presented as [One-Shot GAN](https://openaccess.thecvf.com/content/CVPR2021W/LLID/html/Sushko_One-Shot_GAN_Learning_To_Generate_Samples_From_Single_Images_and_CVPRW_2021_paper.html)). The code allows the users to
reproduce and extend the results reported in the studies. Please cite the papers when reporting, reproducing or extending the results.


<p align="center">
<img src="readme/teaser_osmis.jpg" >
</p>


## Setup

The code is tested for Python 3.8 and the packages listed in [environment.yml](environment.yml).
You can simply install all the packages by running the following command:

```
conda env create --file environment.yml
conda activate osmis
```

**Note**: due to licencing issues, the code does not provide the implementation of differentiable augmentation (DA) from [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch). To fully reproduce the paper results, the users are encouraged to incorporate DA into this repository using quidelines from [here](https://github.com/boschresearch/one-shot-synthesis/issues/1#issuecomment-1140862976). In this case, please install the additional packages using the instructions in the original DA repository and place the DA implementation in ```core/differentiable_augmentation/```. Alternatively, the code can be run with a provided reduced DA implementation by using the ```--use_kornia_augm --prob_augm 0.7``` options. 


## Preparing the data

Just create a name for your input image-mask pair and copy the training data under  ```datasets/$name/```. If you don't have a segmentation mask, or you simply want to run the code without it, you can also have a single image or video without masks. In this case, or in case you activate the ```--use_masks``` option, the code will automatically launch SIV-GAN training only on images. Examples of dataset structures can be found in [./datasets](datasets).


## Training the model

To train the model, you can use the following command:

```
python train.py --exp_name test_run --dataset_name example_image --num_epochs 150000 --max_size 330
```

In this command, the ```--exp_name``` parameter gives each experiment a unique identifier. This way, the intermediate results will be tracked in the folder ```./checkpoints/$exp_name```.  The ```--max_size``` parameter controls the output resolution. Based on this parameter and the original image aspect ratio, the code will automatically construct a recommended model configuration,. The full list of the configuration options can be found in [config.py](config.py).

If your experiment was interrupted unexpectedly, you can continue training by running

```
python train.py --exp_name test_run --continue_train
```

## Generating images after training

After training the model, you can generate images with the following command:

```
python test.py --exp_name test_run --which_epoch 150000 
```

You can select the epoch to evaluate via ```--which_epoch```. Usually, using later epochs leads to higher overal image quality, but decreased diversity (and vice versa).
By default, 100 image-mask pairs will be generated and saved in ```./checkpoints/$exp_name/evaluation/$which_epoch```. The number of generated pairs can be controlled via the ```--num_generated``` parameter.

## Evaluation

To compute metrics for the set of generated images, execute

```
python evaluate.py --exp_name test_run --epoch 150000 
```

The computed SIFID, LPIPS, mIoU, and Distance to train. metrics will be saved in ```./checkpoints/$exp_name/metrics``` as numpy files. To evaluate SIFID at different InceptionV3 layers, you can call the script with a ```--sifid_all_layers``` option. 



# Additional information


## Video Summary of One-Shot GAN
[![video summary](readme/youtube.png)](https://www.youtube.com/watch?v=nsuC2sQvGfk)

## Citation
If you use this code please cite

```
@article{sushko2021generating,
  title={Generating Novel Scene Compositions from Single Images and Videos},
  author={Sushko, Vadim and Zhang, Dan and Gall, Juergen and Khoreva, Anna},
  journal={arXiv preprint:2103.13389},
  year={2021}
}
```
```
@article{sushko2022one,
  title={One-Shot Synthesis of Images and Segmentation Masks},
  author={Sushko, Vadim and Zhang, Dan and Gall, Juergen and Khoreva, Anna},
  journal={arXiv preprint:2209.07547},
  year={2022}
}
```

## License

This project is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in this project, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).

## Purpose of the project

This software is a research prototype, solely developed for and published as
part of the publication cited above. It will neither be
maintained nor monitored in any way.

## Contact
Please feel free to open an issue or contact personally via email, using   
vad221@gmail.com  
vadim.sushko@bosch.com  

## 训练流程
---------------
Step0:输入图像，设计为3个尺度 5*3*360*360 5*3*180*180 5*3*90*90 

---------------
Step1: 生成器，输入Noise，生成器输出

	a. 7个维度的Feature(5*256*5*5 5*256*10*10 ….. 5*32*320*320)，
	b. 将Feature转化为Image(5*3*5*5, 5*3*10*10, ……., 5*3*2*2)
	c. 生成Mask

Step2：扩充模块，对生成器生成的图像进行数据扩充

Step3:  判别器（Noise）

	a. 对输入的图像5*3*320*320 5*3*160*160  5*3*80*80 进行特征提取，得到lowlevel的特征
	b. 对lowlevel最后40*40的输入的特征进行通道维度和空间维度操作
	c. Content 空间维度卷积操作，得到N*1，卷积得到1*1、1*1、1*1、1*1的Tensor
	d. Layout 通道维度卷积操作，得到5*1*20*20 、5*1*10*10 、5*1*5*5、 5*1*1*2

Step4：计算生成器的Loss（Noise）

	a. Lowlevel loss 3个尺度分别与1计算BCE_Loss\
	b. Content Loss 4个尺度分别与1计算BCE_Loss
	c. LayOut  Loss 4个尺度分别与1计算BCE_Loss
	d, 多样性Loss 7个空间尺度 5个样本，计算相同尺度下两两的L1 Loss
--------------- 
Step5：判别器(真实样本)

	a. 对输入的图像5*3*320*320 5*3*160*160  5*3*80*80 进行特征提取，得到lowlevel的特征
	b. 对lowlevel最后40*40的输入的特征进行通道维度和空间维度操作
	c. Content 空间维度卷积操作，得到N*1，样本之间的信息随机交换/丢弃，得到1*1、1*1、1*1、1*1的Tensor
	d. Layout 通道维度卷积操作，样本之间在空间维度进行信息交换，得到5*1*20*20 、5*1*10*10 、5*1*5*5、 5*1*1*2

Step6：计算判别器的Loss(真实样本)

	a. Lowlevel loss 3个尺度分别与1计算BCE_Loss\
	b. Content Loss 4个尺度分别与1计算BCE_Loss
	c. LayOut  Loss 4个尺度分别与1计算BCE_Loss
	d. 多样性Loss 7个空间尺度 5个样本，计算相同尺度下两两的L1 Loss
	
Step7：判别器(生成样本)

	a. 对输入的图像5*3*320*320 5*3*160*160  5*3*80*80 进行特征提取，得到lowlevel的特征
	b. 对lowlevel最后40*40的输入的特征进行通道维度和空间维度操作
	c. Content 空间维度卷积操作，得到N*1，得到1*1、1*1、1*1、1*1的Tensor
	d. Layout 通道维度卷积操作，得到5*1*20*20 、5*1*10*10 、5*1*5*5、 5*1*1*2

Step8：计算判别器的Loss(生成样本)

	a. Lowlevel loss 3个尺度分别与1计算BCE_Loss\
	b. Content Loss 4个尺度分别与1计算BCE_Loss
	c. LayOut  Loss 4个尺度分别与1计算BCE_Loss
	d. 多样性Loss 7个空间尺度 5个样本，计算相同尺度下两两的L1 Loss
---------------