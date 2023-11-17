from .AugmentPipe_kornia import AugmentPipe_kornia
from .AugmentPipe_kornia_New import AugmentPipe_kornia_New


class augment_pipe():
    def __init__(self, opt):
        if opt.use_kornia_augm:
            self.augment_func = AugmentPipe_kornia(opt.prob_augm, opt.use_masks).to(opt.device)
            #self.augment_func = AugmentPipe_kornia_New(opt.prob_augm, opt.use_masks).to(opt.device)
        else:
            raise NotImplementedError("Please install Differentiable Augmentation (DA) using the instructions from https://github.com/NVlabs/stylegan2-ada-pytorch")

    def __call__(self, batch, real=True):
        return self.augment_func(batch)


