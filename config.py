import argparse
import pickle
import os


def read_arguments(train=True):
    parser = get_arguments()
    opt    = parser.parse_args()
    continue_train = opt.continue_train
    num_epochs     = opt.num_epochs
    which_epoch    = opt.which_epoch
    if opt.continue_train or not train:
        update_options_from_file(opt, parser)
        
    opt = parser.parse_args()
    opt.device = "cpu" if opt.cpu else "cuda:{}".format(opt.device_ids)
    opt.phase  = 'train' if train else 'test'
    opt.continue_train = continue_train
    opt.num_epochs     = num_epochs
    opt.debug_folds    = os.path.join(opt.checkpoints_dir, opt.exp_name, "Debug")
    if train:
        opt.continue_epoch = 0 if not opt.continue_train else load_iter(opt)
    else:
        opt.continue_epoch = which_epoch
    if train:
        save_options(opt, parser)
    return opt


def get_arguments():
    parser = argparse.ArgumentParser()

    # basics:
    # Tim.Zhang
    parser.add_argument('--exp_name', help='experiment name for trained folder', default="mvtec_carpet_mask_aug1")
    parser.add_argument('--cpu',      action='store_true', help='run on cpu')
    parser.add_argument('--dataroot', help='location of datasets', default='datasets/')
    parser.add_argument('--checkpoints_dir', help='location of experiments', default='checkpoints/')
    parser.add_argument('--device_ids', type=int, default=0, help='gpu ids: e.g. 0 0 1 2 2 2 3 3 3 [default: 0]')
    
    # Tim.Zhang
    parser.add_argument('--dataset_name', help='dataset name', default='carpet')     
    
    # Tim.Zhang
    parser.add_argument('--num_epochs', type=int, default=100000, help='number of epochs')
    
    # Tim.Zhang
    parser.add_argument('--max_size',   type=int, default=224, help='limit image size in max dimension')
    
    parser.add_argument('--continue_train', type=int,  default=0,      help='continue training of a previous checkpoint?')
    parser.add_argument('--which_epoch',    type=int,  default=90000,  help='which epoch to use for evaluation')
    parser.add_argument('--num_generated',  type=int,  default=100,    help='how many images to generate for evaluation')

    # regime
    parser.add_argument('--use_masks',       type=int, default=1, help='use the regime without segmentation masks')
    parser.add_argument('--use_kornia_augm', type=int, default=1, help='use an older version of differentiable augm')
    parser.add_argument('--use_read_augm',   type=int, default=1, help='use augm for reading images')

    # training:
    parser.add_argument('--batch_size',  type=int,   default=5,      help='batch_size')
    parser.add_argument('--noise_dim',   type=int,   default=64,     help='dimension of noise vector')
    parser.add_argument('--lr_g',        type=float, default=0.0002, help='generator learning rate')
    parser.add_argument('--lr_d',        type=float, default=0.0002, help='discriminator learning rate')
    parser.add_argument('--beta1',       type=float, default=0.5,    help='beta1 for adam')
    parser.add_argument('--beta2',       type=float, default=0.999,  help='beta2 for adam')
    parser.add_argument('--loss_mode',   type=str,   default="bce",  help='which GAN loss (wgan|hinge|bce)')
    parser.add_argument('--seed',        type=int,   default=22,     help='which randomm seed to use')
    parser.add_argument('--use_DR',      type=int,   default=1,      help='deactivate Diversity Regularization?')
    parser.add_argument('--prob_augm',   type=float, default=0.3,    help='probability of data augmentation')
    parser.add_argument('--lambda_DR',   type=float, default=0.15,   help='lambda for DR')
    parser.add_argument('--prob_FA_con', type=float, default=0.4,    help='probability of content FA')
    parser.add_argument('--prob_FA_lay', type=float, default=0.4,    help='probability of layout FA')
    parser.add_argument('--use_EMA',     type=int,   default=1,      help='deactivate exponential moving average of G weights?')
    parser.add_argument('--EMA_decay',   type=float, default=0.9999,   help='decay for exponential moving averages')
    parser.add_argument('--bernoulli_warmup', type=int, default=15000, help='epochs for soft_mask bernoulli warmup')

    # architecture
    parser.add_argument('--norm_G', help='which norm to use in generator     (None|batch|instance)', default="none")
    parser.add_argument('--norm_D', help='which norm to use in discriminator (None|batch|instance)', default="none")
    parser.add_argument('--ch_G', type=float, help='channel multiplier for G blocks', default=32)
    parser.add_argument('--ch_D', type=float, help='channel multiplier for D blocks', default=32)
    parser.add_argument('--num_blocks_d', type=int, help='Discriminator blocks number. 0 -> use recommended default', default=0)
    parser.add_argument('--num_blocks_d0', type=int, help='Num of D_low-level blocks. 0 -> use recommended default', default=0)

    # stats tracking
    parser.add_argument('--freq_save_loss', type=int, help='frequency of loss plot updates',       default=500)
    parser.add_argument('--freq_print',     type=int, help='frequency of saving images and timer', default=500)
    parser.add_argument('--freq_save_ckpt', type=int, help='frequency of saving checkpoints',      default=5000)
    return parser


def update_options_from_file(opt, parser):
    file_name = os.path.join(opt.checkpoints_dir, opt.exp_name, "opt.pkl")
    new_opt = pickle.load(open(file_name, 'rb'))
    for k, v in sorted(vars(opt).items()):
        if hasattr(new_opt, k) and v != getattr(new_opt, k):
            new_val = getattr(new_opt, k)
            parser.set_defaults(**{k: new_val})
    return parser


def load_iter(opt):
    with open(os.path.join(opt.checkpoints_dir, opt.exp_name, "models/latest_epoch.txt"), "r") as f:
        res = int(f.read())
        return res


def save_options(opt, parser):
    path_name = os.path.join(opt.checkpoints_dir, opt.exp_name)
    os.makedirs(path_name, exist_ok=True)
    with open(path_name + '/opt.txt', 'wt') as opt_file:
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

    with open(path_name + '/opt.pkl', 'wb') as opt_file:
        pickle.dump(opt, opt_file)