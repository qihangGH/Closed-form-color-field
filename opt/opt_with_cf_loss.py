import torch
import torch.cuda
import torch.optim
import torch.nn.functional as F
import svox2
import json
import imageio
import os
from os import path
import shutil
import gc
import numpy as np
import math
import argparse
import cv2
from util.dataset import datasets
from util.util import Timing, get_expon_lr_func, generate_dirs_equirect, viridis_cmap, get_nearest_pose_ids
from util import config_util

from warnings import warn
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from typing import NamedTuple, Optional, Union
from util.util import calc_color_metric

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
config_util.define_common_args(parser)

group = parser.add_argument_group("general")
group.add_argument('--train_dir', '-t', type=str, default=
'ckpt/test',
                   help='checkpoint and logging directory')

group.add_argument('--reso',
                   type=str,
                   default=
                   "[[256, 256, 256], [512, 512, 512]]",
                   help='List of grid resolution (will be evaled as json);'
                        'resamples to the next one every upsamp_every iters, then ' +
                        'stays at the last one; ' +
                        'should be a list where each item is a list of 3 ints or an int')
group.add_argument('--upsamp_every', type=int, default=
3 * 12800,
                   help='upsample the grid every x iters')
group.add_argument('--init_iters', type=int, default=
0,
                   help='do not upsample for first x iters')
group.add_argument('--upsample_density_add', type=float, default=
0.0,
                   help='add the remaining density by this amount when upsampling')

group.add_argument('--basis_type',
                   choices=['sh', '3d_texture', 'mlp'],
                   default='sh',
                   help='Basis function type')

group.add_argument('--basis_reso', type=int, default=32,
                   help='basis grid resolution (only for learned texture)')
group.add_argument('--sh_dim', type=int, default=
9,
                   help='SH/learned basis dimensions')

group.add_argument('--mlp_posenc_size', type=int, default=4,
                   help='Positional encoding size if using MLP basis; 0 to disable')
group.add_argument('--mlp_width', type=int, default=32, help='MLP width if using MLP basis')

group.add_argument('--background_nlayers', type=int, default=0,  # 32,
                   help='Number of background layers (0=disable BG model)')
group.add_argument('--background_reso', type=int, default=512, help='Background resolution')

group = parser.add_argument_group("optimization")
group.add_argument('--n_iters', type=int, default=10 * 12800, help='total number of iters to optimize for')
group.add_argument('--batch_size', type=int, default=
5000,
                   help='batch size')

group.add_argument('--sigma_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Density optimizer")
group.add_argument('--lr_sigma', type=float, default=3e1, help='SGD/rmsprop lr for sigma')
group.add_argument('--lr_sigma_final', type=float, default=5e-2)
group.add_argument('--lr_sigma_decay_steps', type=int, default=250000)
group.add_argument('--lr_sigma_delay_steps', type=int, default=15000,
                   help="Reverse cosine steps (0 means disable)")
group.add_argument('--lr_sigma_delay_mult', type=float, default=1e-2)  # 1e-4)#1e-4)

group.add_argument('--sh_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="SH optimizer")
group.add_argument('--lr_sh', type=float, default=
1e-2,
                   help='SGD/rmsprop lr for SH')
group.add_argument('--lr_sh_final', type=float,
                   default=
                   5e-6
                   )
group.add_argument('--lr_sh_decay_steps', type=int, default=250000)
group.add_argument('--lr_sh_delay_steps', type=int, default=0, help="Reverse cosine steps (0 means disable)")
group.add_argument('--lr_sh_delay_mult', type=float, default=1e-2)

group.add_argument('--lr_fg_begin_step', type=int, default=0, help="Foreground begins training at given step number")

# BG LRs
group.add_argument('--bg_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Background optimizer")
group.add_argument('--lr_sigma_bg', type=float, default=3e0,
                   help='SGD/rmsprop lr for background')
group.add_argument('--lr_sigma_bg_final', type=float, default=3e-3,
                   help='SGD/rmsprop lr for background')
group.add_argument('--lr_sigma_bg_decay_steps', type=int, default=250000)
group.add_argument('--lr_sigma_bg_delay_steps', type=int, default=0, help="Reverse cosine steps (0 means disable)")
group.add_argument('--lr_sigma_bg_delay_mult', type=float, default=1e-2)

group.add_argument('--lr_color_bg', type=float, default=1e-1,
                   help='SGD/rmsprop lr for background')
group.add_argument('--lr_color_bg_final', type=float, default=5e-6,  # 1e-4,
                   help='SGD/rmsprop lr for background')
group.add_argument('--lr_color_bg_decay_steps', type=int, default=250000)
group.add_argument('--lr_color_bg_delay_steps', type=int, default=0, help="Reverse cosine steps (0 means disable)")
group.add_argument('--lr_color_bg_delay_mult', type=float, default=1e-2)
# END BG LRs

group.add_argument('--basis_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Learned basis optimizer")
group.add_argument('--lr_basis', type=float, default=  # 2e6,
1e-6,
                   help='SGD/rmsprop lr for SH')
group.add_argument('--lr_basis_final', type=float,
                   default=
                   1e-6
                   )
group.add_argument('--lr_basis_decay_steps', type=int, default=250000)
group.add_argument('--lr_basis_delay_steps', type=int, default=0,  # 15000,
                   help="Reverse cosine steps (0 means disable)")
group.add_argument('--lr_basis_begin_step', type=int, default=0)  # 4 * 12800)
group.add_argument('--lr_basis_delay_mult', type=float, default=1e-2)

group.add_argument('--rms_beta', type=float, default=0.95, help="RMSProp exponential averaging factor")

group.add_argument('--print_every', type=int, default=20, help='print every')
group.add_argument('--save_every', type=int, default=5,
                   help='save every x epochs')
group.add_argument('--eval_every', type=int, default=1,
                   help='evaluate every x epochs')

group.add_argument('--init_sigma', type=float,
                   default=0.1,
                   help='initialization sigma')
group.add_argument('--init_sigma_bg', type=float,
                   default=0.1,
                   help='initialization sigma (for BG)')

# Extra logging
group.add_argument('--log_mse_image', action='store_true', default=False)
group.add_argument('--log_depth_map', action='store_true', default=False)
group.add_argument('--log_depth_map_use_thresh', type=float, default=None,
                   help="If specified, uses the Dex-neRF version of depth with given thresh; else returns expected term")

group = parser.add_argument_group("misc experiments")
group.add_argument('--thresh_type',
                   choices=["weight", "sigma"],
                   default="weight",
                   help='Upsample threshold type')
group.add_argument('--weight_thresh', type=float,
                   default=0.0005 * 512,
                   help='Upsample weight threshold; will be divided by resulting z-resolution')
group.add_argument('--density_thresh', type=float,
                   default=5.0,
                   help='Upsample sigma threshold')
group.add_argument('--background_density_thresh', type=float,
                   default=1.0 + 1e-9,
                   help='Background sigma threshold for sparsification')
group.add_argument('--max_grid_elements', type=int,
                   default=44_000_000,
                   help='Max items to store after upsampling '
                        '(the number here is given for 22GB memory)')

group.add_argument('--tune_mode', action='store_true', default=False,
                   help='hypertuning mode (do not save, for speed)')
group.add_argument('--tune_nosave', action='store_true', default=False,
                   help='do not save any checkpoint even at the end')

group = parser.add_argument_group("losses")
# Foreground TV
group.add_argument('--lambda_tv', type=float, default=1e-5)
group.add_argument('--tv_sparsity', type=float, default=0.01)
group.add_argument('--tv_logalpha', action='store_true', default=False,
                   help='Use log(1-exp(-delta * sigma)) as in neural volumes')

group.add_argument('--lambda_tv_sh', type=float, default=1e-3)
group.add_argument('--tv_sh_sparsity', type=float, default=0.01)

group.add_argument('--lambda_tv_lumisphere', type=float, default=0.0)  # 1e-2)#1e-3)
group.add_argument('--tv_lumisphere_sparsity', type=float, default=0.01)
group.add_argument('--tv_lumisphere_dir_factor', type=float, default=0.0)

group.add_argument('--tv_decay', type=float, default=1.0)

group.add_argument('--lambda_l2_sh', type=float, default=0.0)  # 1e-4)
group.add_argument('--tv_early_only', type=int, default=1,
                   help="Turn off TV regularization after the first split/prune")

group.add_argument('--tv_contiguous', type=int, default=1,
                   help="Apply TV only on contiguous link chunks, which is faster")
# End Foreground TV

group.add_argument('--lambda_sparsity', type=float, default=
0.0,
                   help="Weight for sparsity loss as in SNeRG/PlenOctrees " +
                        "(but applied on the ray)")
group.add_argument('--lambda_beta', type=float, default=
0.0,
                   help="Weight for beta distribution sparsity loss as in neural volumes")

# Background TV
group.add_argument('--lambda_tv_background_sigma', type=float, default=1e-2)
group.add_argument('--lambda_tv_background_color', type=float, default=1e-2)

group.add_argument('--tv_background_sparsity', type=float, default=0.01)
# End Background TV

# Basis TV
group.add_argument('--lambda_tv_basis', type=float, default=0.0,
                   help='Learned basis total variation loss')
# End Basis TV
group.add_argument('--weight_decay_sigma', type=float, default=1.0)
group.add_argument('--weight_decay_sh', type=float, default=1.0)

group.add_argument('--lr_decay', action='store_true', default=True)

group.add_argument('--n_train', type=int, default=None, help='Number of training images. Defaults to use all avaiable.')

group.add_argument('--nosphereinit', action='store_true', default=False,
                   help='do not start with sphere bounds (please do not use for 360)')

group = parser.add_argument_group("cf loss")
group.add_argument('--cf_loss_ray_frac', default=0.005, type=float,
                   help='The fractions of batch size to calculate the cf loss'
                   )
group.add_argument("--lambda_cf_loss", default=10, type=float, help='The weight of cf loss')
group.add_argument('--estimate_color_every', default=
None,
                   help='The frequency of color estimation for different resolutions.'
                   )
group.add_argument('--cf_include_cur', default=True, action="store_false")

args = parser.parse_args()
config_util.maybe_merge_config_file(args)

print("Density tv:", args.lambda_tv)
print("SH tv:", args.lambda_tv_sh)

if args.estimate_color_every is None:
    estimate_color_every = [1 for _ in range(len(json.loads(args.reso)))]
else:
    estimate_color_every = [int(i) for i in args.estimate_color_every.split('_')]
assert 0. <= args.cf_loss_ray_frac <= 1.
print("cf_loss_ray_frac", args.cf_loss_ray_frac)
print("lambda_cf_loss", args.lambda_cf_loss)
print("estimate_color_every", estimate_color_every)

assert args.lr_sigma_final <= args.lr_sigma, "lr_sigma must be >= lr_sigma_final"
assert args.lr_sh_final <= args.lr_sh, "lr_sh must be >= lr_sh_final"
assert args.lr_basis_final <= args.lr_basis, "lr_basis must be >= lr_basis_final"

os.makedirs(args.train_dir, exist_ok=True)
summary_writer = SummaryWriter(args.train_dir)

reso_list = json.loads(args.reso)
reso_id = 0

with open(path.join(args.train_dir, 'args.json'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)
    # Changed name to prevent errors
    shutil.copyfile(__file__, path.join(args.train_dir, 'opt_frozen.py'))

torch.manual_seed(20200823)
np.random.seed(20200823)

factor = 1
dset = datasets[args.dataset_type](
    args.data_dir,
    split="train",
    device=device,
    factor=factor,
    n_images=args.n_train,
    **config_util.build_data_options(args))

c2ws = dset.c2w.to(device=device).float()
w2cs = torch.inverse(c2ws).contiguous()
# For DTU and NeRF Synthetic intrinsics.size(1) < 9
intrinsics = []
for i in range(c2ws.shape[0]):
    intrin = [
        dset.intrins.get('fx', i),
        dset.intrins.get('fy', i),
        dset.intrins.get('cx', i),
        dset.intrins.get('cy', i),
        dset.get_image_size(i)[1],
        dset.get_image_size(i)[0]
    ]
    if dset.ndc_coeffs[0] != -1:
        intrin.extend([
            2 * dset.intrins.get('fx', i) / dset.get_image_size(i)[1],
            2 * dset.intrins.get('fy', i) / dset.get_image_size(i)[0],
            1.0
        ])
    intrinsics.append(torch.tensor(intrin))
intrinsics = torch.stack(intrinsics).to(device=device).float()
images = dset.gt.to(device=device)

# Sort cam pose
sorted_ind = []
c2ws_np = dset.c2w.numpy()
for idx, c2w in enumerate(c2ws_np):
    # Include the target camera
    sorted_idx = get_nearest_pose_ids(
        c2w, c2ws_np, num_select=len(c2ws_np), tar_id=-1,
        angular_dist_method='vector', scene_center=(0, 0, 0)
    )
    sorted_ind.append(sorted_idx)
sorted_ind = np.stack(sorted_ind)
sorted_ind = torch.from_numpy(sorted_ind).long()  # [num_cams, num_cams]
# sorted_ind = sorted_ind[:, None].repeat([1, dset.h * dset.w, 1]).reshape([-1, len(c2ws_np)])
unshuffled_ray_o = dset.rays.origins.reshape([len(c2ws_np), dset.h * dset.w, -1])
unshuffled_ray_d = dset.rays.dirs.reshape([len(c2ws_np), dset.h * dset.w, -1])
unshuffled_ray_d = unshuffled_ray_d / torch.norm(unshuffled_ray_d, dim=-1, keepdim=True)
unshuffled_gt = dset.gt.reshape([len(c2ws_np), dset.h * dset.w, -1])
num_near_start = 1
num_near_end = len(c2ws_np)
print(f"num_near_start: {num_near_start}, num_near_end: {num_near_end}")

if args.background_nlayers > 0 and not dset.should_use_background:
    warn('Using a background model for dataset type ' + str(type(dset)) + ' which typically does not use background')

dset_test = datasets[args.dataset_type](
    args.data_dir, split="test", **config_util.build_data_options(args))

global_start_time = datetime.now()

grid = svox2.SparseGrid(reso=reso_list[reso_id],
                        center=dset.scene_center,
                        radius=dset.scene_radius,
                        use_sphere_bound=dset.use_sphere_bound and not args.nosphereinit,
                        basis_dim=args.sh_dim,
                        use_z_order=True,
                        device=device,
                        basis_reso=args.basis_reso,
                        basis_type=svox2.__dict__['BASIS_TYPE_' + args.basis_type.upper()],
                        mlp_posenc_size=args.mlp_posenc_size,
                        mlp_width=args.mlp_width,
                        background_nlayers=args.background_nlayers,
                        background_reso=args.background_reso)

# DC -> gray; mind the SH scaling!
grid.sh_data.data[:] = 0.0
grid.density_data.data[:] = 0.0 if args.lr_fg_begin_step > 0 else args.init_sigma

if grid.use_background:
    grid.background_data.data[..., -1] = args.init_sigma_bg
    #  grid.background_data.data[..., :-1] = 0.5 / svox2.utils.SH_C0

optim_basis_mlp = None

if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
    grid.reinit_learned_bases(init_type='sh')

elif grid.basis_type == svox2.BASIS_TYPE_MLP:
    # MLP!
    optim_basis_mlp = torch.optim.Adam(
        grid.basis_mlp.parameters(),
        lr=args.lr_basis
    )

grid.requires_grad_(True)
config_util.setup_render_opts(grid.opt, args)
print('Render options', grid.opt)

gstep_id_base = 0

resample_cameras = [
    svox2.Camera(c2w.to(device=device),
                 dset.intrins.get('fx', i),
                 dset.intrins.get('fy', i),
                 dset.intrins.get('cx', i),
                 dset.intrins.get('cy', i),
                 width=dset.get_image_size(i)[1],
                 height=dset.get_image_size(i)[0],
                 ndc_coeffs=dset.ndc_coeffs) for i, c2w in enumerate(dset.c2w)
]
ckpt_path = path.join(args.train_dir, 'ckpt')

lr_sigma_func = get_expon_lr_func(args.lr_sigma, args.lr_sigma_final, args.lr_sigma_delay_steps,
                                  args.lr_sigma_delay_mult, args.lr_sigma_decay_steps)
lr_sh_func = get_expon_lr_func(args.lr_sh, args.lr_sh_final, args.lr_sh_delay_steps,
                               args.lr_sh_delay_mult, args.lr_sh_decay_steps)
lr_basis_func = get_expon_lr_func(args.lr_basis, args.lr_basis_final, args.lr_basis_delay_steps,
                                  args.lr_basis_delay_mult, args.lr_basis_decay_steps)
lr_sigma_bg_func = get_expon_lr_func(args.lr_sigma_bg, args.lr_sigma_bg_final, args.lr_sigma_bg_delay_steps,
                                     args.lr_sigma_bg_delay_mult, args.lr_sigma_bg_decay_steps)
lr_color_bg_func = get_expon_lr_func(args.lr_color_bg, args.lr_color_bg_final, args.lr_color_bg_delay_steps,
                                     args.lr_color_bg_delay_mult, args.lr_color_bg_decay_steps)
lr_sigma_factor = 1.0
lr_sh_factor = 1.0
lr_basis_factor = 1.0

last_upsamp_step = args.init_iters

if args.enable_random:
    warn("Randomness is enabled for training (normal for LLFF & scenes with background)")

epoch_id = -1
while True:
    dset.shuffle_rays()
    epoch_id += 1
    epoch_size = dset.rays.origins.size(0)
    batches_per_epoch = (epoch_size - 1) // args.batch_size + 1

    # Test
    def eval_step(dset_test, prefix):
        # Put in a function to avoid memory leak
        print('Eval step')
        with torch.no_grad():
            stats_test = {'psnr': 0.0, 'mse': 0.0}

            # Standard set
            N_IMGS_TO_EVAL = min(20 if epoch_id > 0 else 5, dset_test.n_images)
            N_IMGS_TO_SAVE = N_IMGS_TO_EVAL  # if not args.tune_mode else 1
            img_eval_interval = dset_test.n_images // N_IMGS_TO_EVAL
            img_save_interval = (N_IMGS_TO_EVAL // N_IMGS_TO_SAVE)
            img_ids = range(0, dset_test.n_images, img_eval_interval)

            n_images_gen = 0
            for i, img_id in tqdm(enumerate(img_ids), total=len(img_ids)):
                c2w = dset_test.c2w[img_id].to(device=device)
                cam = svox2.Camera(c2w,
                                   dset_test.intrins.get('fx', img_id),
                                   dset_test.intrins.get('fy', img_id),
                                   dset_test.intrins.get('cx', img_id),
                                   dset_test.intrins.get('cy', img_id),
                                   width=dset_test.get_image_size(img_id)[1],
                                   height=dset_test.get_image_size(img_id)[0],
                                   ndc_coeffs=dset_test.ndc_coeffs)
                rgb_pred_test = grid.volume_render_image(cam, use_kernel=True)
                rgb_gt_test = dset_test.gt[img_id].to(device=device)
                all_mses = ((rgb_gt_test - rgb_pred_test) ** 2).cpu()
                if i % img_save_interval == 0:
                    img_pred = rgb_pred_test.cpu()
                    img_pred.clamp_max_(1.0)
                    summary_writer.add_image(f'{prefix}/image_{img_id:04d}',
                                             img_pred, global_step=gstep_id_base, dataformats='HWC')
                    if args.log_mse_image:
                        mse_img = all_mses / all_mses.max()
                        summary_writer.add_image(f'{prefix}/mse_map_{img_id:04d}',
                                                 mse_img, global_step=gstep_id_base, dataformats='HWC')
                    if args.log_depth_map:
                        depth_img = grid.volume_render_depth_image(cam,
                                                                   args.log_depth_map_use_thresh if
                                                                   args.log_depth_map_use_thresh else None
                                                                   )
                        depth_img = viridis_cmap(depth_img.cpu())
                        summary_writer.add_image(f'{prefix}/depth_map_{img_id:04d}',
                                                 depth_img,
                                                 global_step=gstep_id_base, dataformats='HWC')

                rgb_pred_test = rgb_gt_test = None
                mse_num: float = all_mses.mean().item()
                psnr = -10.0 * math.log10(mse_num)
                if math.isnan(psnr):
                    print('NAN PSNR', i, img_id, mse_num)
                    assert False
                stats_test['mse'] += mse_num
                stats_test['psnr'] += psnr
                n_images_gen += 1

            if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE or \
                    grid.basis_type == svox2.BASIS_TYPE_MLP:
                # Add spherical map visualization
                EQ_RESO = 256
                eq_dirs = generate_dirs_equirect(EQ_RESO * 2, EQ_RESO)
                eq_dirs = torch.from_numpy(eq_dirs).to(device=device).view(-1, 3)

                if grid.basis_type == svox2.BASIS_TYPE_MLP:
                    sphfuncs = grid._eval_basis_mlp(eq_dirs)
                else:
                    sphfuncs = grid._eval_learned_bases(eq_dirs)
                sphfuncs = sphfuncs.view(EQ_RESO, EQ_RESO * 2, -1).permute([2, 0, 1]).cpu().numpy()

                stats = [(sphfunc.min(), sphfunc.mean(), sphfunc.max())
                         for sphfunc in sphfuncs]
                sphfuncs_cmapped = [viridis_cmap(sphfunc) for sphfunc in sphfuncs]
                for im, (minv, meanv, maxv) in zip(sphfuncs_cmapped, stats):
                    cv2.putText(im, f"{minv=:.4f} {meanv=:.4f} {maxv=:.4f}", (10, 20),
                                0, 0.5, [255, 0, 0])
                sphfuncs_cmapped = np.concatenate(sphfuncs_cmapped, axis=0)
                summary_writer.add_image(f'{prefix}/spheric',
                                         sphfuncs_cmapped, global_step=gstep_id_base, dataformats='HWC')
                # END add spherical map visualization

            stats_test['mse'] /= n_images_gen
            stats_test['psnr'] /= n_images_gen
            for stat_name in stats_test:
                summary_writer.add_scalar(f'{prefix}/' + stat_name,
                                          stats_test[stat_name], global_step=gstep_id_base)
            summary_writer.add_scalar('epoch_id', float(epoch_id), global_step=gstep_id_base)
            print('eval stats:', stats_test)


    if epoch_id % max(factor, args.eval_every) == 0:  # and (epoch_id > 0 or not args.tune_mode):
        with torch.no_grad():
            if epoch_id > 0:
                print('Start estimate color')
                with Timing("color estimated"):
                    color_estimated, color_weight, color_res = \
                        grid.estimate_color_metric(c2ws, w2cs, intrinsics, images)
                print(f"Shape of the estimated color: {color_estimated.cpu().shape}")

                store_sh = grid.sh_data.data.clone()
                grid.sh_data.data[:, :] = color_estimated[:, :]
                eval_step(dset_test, "cf_test")
                grid.sh_data.data[:, :] = store_sh.clone()
                if args.cf_loss_ray_frac > 0:
                    eval_step(dset_test, "test")

                metric, _, _ = calc_color_metric(grid, color_res, color_weight)
                imrc = -10 * np.log10(metric.detach().cpu().numpy())
                print(f'IMRC: {imrc:.2f}')
                summary_writer.add_scalar('imrc', float(imrc), global_step=gstep_id_base)
            else:
                if args.cf_loss_ray_frac > 0:
                    eval_step(dset_test, "test")
        gc.collect()


    def train_step():
        print('Train step')
        pbar = tqdm(enumerate(range(0, epoch_size, args.batch_size)), total=batches_per_epoch)
        stats = {"mse": 0.0, "psnr": 0.0, "invsqr_mse": 0.0, "mse_cf": 0.0, "psnr_cf": 0.0}
        for iter_id, batch_begin in pbar:
            gstep_id = iter_id + gstep_id_base
            if args.lr_fg_begin_step > 0 and gstep_id == args.lr_fg_begin_step:
                grid.density_data.data[:] = args.init_sigma
            lr_sigma = lr_sigma_func(gstep_id) * lr_sigma_factor
            lr_sh = lr_sh_func(gstep_id) * lr_sh_factor
            lr_basis = lr_basis_func(gstep_id - args.lr_basis_begin_step) * lr_basis_factor
            lr_sigma_bg = lr_sigma_bg_func(gstep_id - args.lr_basis_begin_step) * lr_basis_factor
            lr_color_bg = lr_color_bg_func(gstep_id - args.lr_basis_begin_step) * lr_basis_factor
            if not args.lr_decay:
                lr_sigma = args.lr_sigma * lr_sigma_factor
                lr_sh = args.lr_sh * lr_sh_factor
                lr_basis = args.lr_basis * lr_basis_factor

            batch_end = min(batch_begin + args.batch_size, epoch_size)
            batch_slice = slice(batch_begin, batch_end)
            batch_origins = dset.rays.origins[batch_slice]
            batch_dirs = dset.rays.dirs[batch_slice]
            rgb_gt = dset.rays.gt[batch_slice]
            rays = svox2.Rays(batch_origins, batch_dirs)

            if iter_id % estimate_color_every[reso_id] == 0:
                batch_cf = int((batch_end - batch_begin) * args.cf_loss_ray_frac)
                if args.cf_include_cur:
                    batch_slice_cf = slice(batch_begin, batch_begin + batch_cf)
                    batch_origins_cf = dset.rays.origins[batch_slice_cf]
                    batch_dirs_cf = dset.rays.dirs[batch_slice_cf]
                    rgb_gt_cf = dset.rays.gt[batch_slice_cf]
                    rays_cf = svox2.Rays(batch_origins_cf, batch_dirs_cf)
                    near_cam_ind = torch.ones(len(c2ws_np)).bool()
                else:
                    sel_cam = np.random.choice(np.arange(len(c2ws_np)), 1)[0]
                    sel_rays = np.random.choice(np.arange(dset.h * dset.w), batch_cf, replace=False)
                    rays_cf = svox2.Rays(unshuffled_ray_o[sel_cam, sel_rays].to(device),
                                         unshuffled_ray_d[sel_cam, sel_rays].to(device))
                    rgb_gt_cf = unshuffled_gt[sel_cam, sel_rays].to(device)
                    near_cam_ind = sorted_ind[sel_cam, num_near_start:num_near_end]
                    # assert sel_cam not in near_cam_ind
                with torch.no_grad():
                    # mask the voxels that are used to render `rays_cf`
                    mask_out = grid.get_mask_out(rays_cf, True)
                    # mask_out = grid.get_mask_out(rays_cf)
                    # estimate color of the masked voxels
                    color_estimated = grid.estimate_color(
                        mask_out, c2ws[near_cam_ind], w2cs[near_cam_ind],
                        intrinsics[near_cam_ind], images[near_cam_ind])
                    store_sh = grid.sh_data.data[mask_out].clone()
                    # grid.sh_data.data[:, :] = 0.
                    grid.sh_data.data[mask_out] = color_estimated[:, :]

                # beta and sparsity loss only need to be calculated once
                rgb_pred_cf = grid.volume_render_fused(rays_cf, rgb_gt_cf,
                                                       beta_loss=0.,
                                                       sparsity_loss=0.,
                                                       randomize=args.enable_random)

                with torch.no_grad():
                    # Note that `grid.volume_render_fused` re-initializes the grad indexer,
                    # including both density and sh indexes
                    # store_grad_indexes = grid.sparse_grad_indexer.clone()
                    # store_density_grad = args.lambda_cf_loss * grid.density_data.grad
                    store_density_grad = \
                        args.lambda_cf_loss * estimate_color_every[reso_id] * grid.density_data.grad
                    grid.density_data.grad = None
                    grid.sh_data.grad = None
                    grid.sh_data.data[mask_out] = store_sh

            #  with Timing("volrend_fused"):
            rgb_pred = grid.volume_render_fused(rays, rgb_gt,
                                                beta_loss=args.lambda_beta,
                                                sparsity_loss=args.lambda_sparsity,
                                                randomize=args.enable_random)

            if iter_id % estimate_color_every[reso_id] == 0:
                grid.density_data.grad += store_density_grad
                # grid.sparse_grad_indexer |= store_grad_indexes
                # Note that we do not update grid.sh_sparse_grad_indexer,
                # which means fewer SH coefficients will be updated

                mse_cf = F.mse_loss(rgb_gt_cf, rgb_pred_cf)
                mse_cf_num: float = mse_cf.detach().item()
                if mse_cf_num == 0.:
                    mse_cf_num = 1e-10
                psnr_cf = -10.0 * math.log10(mse_cf_num)

            stats['mse_cf'] += mse_cf_num
            stats['psnr_cf'] += psnr_cf

            #  with Timing("loss_comp"):
            mse = F.mse_loss(rgb_gt, rgb_pred)
            # Stats
            mse_num: float = mse.detach().item()
            psnr = -10.0 * math.log10(mse_num)
            stats['mse'] += mse_num
            stats['psnr'] += psnr
            stats['invsqr_mse'] += 1.0 / mse_num ** 2

            if (iter_id + 1) % args.print_every == 0:
                # Print averaged stats
                pbar.set_description(f'epoch {epoch_id} psnr={psnr:.2f} psnr_cf={psnr_cf:.2f}')
                for stat_name in stats:
                    stat_val = stats[stat_name] / args.print_every
                    summary_writer.add_scalar(stat_name, stat_val, global_step=gstep_id)
                    stats[stat_name] = 0.0
                #  if args.lambda_tv > 0.0:
                #      with torch.no_grad():
                #          tv = grid.tv(logalpha=args.tv_logalpha, ndc_coeffs=dset.ndc_coeffs)
                #      summary_writer.add_scalar("loss_tv", tv, global_step=gstep_id)
                #  if args.lambda_tv_sh > 0.0:
                #      with torch.no_grad():
                #          tv_sh = grid.tv_color()
                #      summary_writer.add_scalar("loss_tv_sh", tv_sh, global_step=gstep_id)
                #  with torch.no_grad():
                #      tv_basis = grid.tv_basis() #  summary_writer.add_scalar("loss_tv_basis", tv_basis, global_step=gstep_id)
                summary_writer.add_scalar("lr_sh", lr_sh, global_step=gstep_id)
                summary_writer.add_scalar("lr_sigma", lr_sigma, global_step=gstep_id)
                if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
                    summary_writer.add_scalar("lr_basis", lr_basis, global_step=gstep_id)
                if grid.use_background:
                    summary_writer.add_scalar("lr_sigma_bg", lr_sigma_bg, global_step=gstep_id)
                    summary_writer.add_scalar("lr_color_bg", lr_color_bg, global_step=gstep_id)

                if args.weight_decay_sh < 1.0:
                    grid.sh_data.data *= args.weight_decay_sigma
                if args.weight_decay_sigma < 1.0:
                    grid.density_data.data *= args.weight_decay_sh

            #  # For outputting the % sparsity of the gradient
            #  indexer = grid.sparse_sh_grad_indexer
            #  if indexer is not None:
            #      if indexer.dtype == torch.bool:
            #          nz = torch.count_nonzero(indexer)
            #      else:
            #          nz = indexer.size()
            #      with open(os.path.join(args.train_dir, 'grad_sparsity.txt'), 'a') as sparsity_file:
            #          sparsity_file.write(f"{gstep_id} {nz}\n")

            # Apply TV/Sparsity regularizers
            if args.lambda_tv > 0.0:
                #  with Timing("tv_inpl"):
                grid.inplace_tv_grad(grid.density_data.grad,
                                     scaling=args.lambda_tv,
                                     sparse_frac=args.tv_sparsity,
                                     logalpha=args.tv_logalpha,
                                     ndc_coeffs=dset.ndc_coeffs,
                                     contiguous=args.tv_contiguous)
            if args.lambda_tv_sh > 0.0:
                #  with Timing("tv_color_inpl"):
                grid.inplace_tv_color_grad(grid.sh_data.grad,
                                           scaling=args.lambda_tv_sh,
                                           sparse_frac=args.tv_sh_sparsity,
                                           ndc_coeffs=dset.ndc_coeffs,
                                           contiguous=args.tv_contiguous)
            if args.lambda_tv_lumisphere > 0.0:
                grid.inplace_tv_lumisphere_grad(grid.sh_data.grad,
                                                scaling=args.lambda_tv_lumisphere,
                                                dir_factor=args.tv_lumisphere_dir_factor,
                                                sparse_frac=args.tv_lumisphere_sparsity,
                                                ndc_coeffs=dset.ndc_coeffs)
            if args.lambda_l2_sh > 0.0:
                grid.inplace_l2_color_grad(grid.sh_data.grad,
                                           scaling=args.lambda_l2_sh)
            if grid.use_background and (args.lambda_tv_background_sigma > 0.0 or args.lambda_tv_background_color > 0.0):
                grid.inplace_tv_background_grad(grid.background_data.grad,
                                                scaling=args.lambda_tv_background_color,
                                                scaling_density=args.lambda_tv_background_sigma,
                                                sparse_frac=args.tv_background_sparsity,
                                                contiguous=args.tv_contiguous)
            if args.lambda_tv_basis > 0.0:
                tv_basis = grid.tv_basis()
                loss_tv_basis = tv_basis * args.lambda_tv_basis
                loss_tv_basis.backward()
            #  print('nz density', torch.count_nonzero(grid.sparse_grad_indexer).item(),
            #        ' sh', torch.count_nonzero(grid.sparse_sh_grad_indexer).item())

            # Manual SGD/rmsprop step
            if gstep_id >= args.lr_fg_begin_step:
                grid.optim_density_step(lr_sigma, beta=args.rms_beta, optim=args.sigma_optim)
                grid.optim_sh_step(lr_sh, beta=args.rms_beta, optim=args.sh_optim)
            if grid.use_background:
                grid.optim_background_step(lr_sigma_bg, lr_color_bg, beta=args.rms_beta, optim=args.bg_optim)
            if gstep_id >= args.lr_basis_begin_step:
                if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
                    grid.optim_basis_step(lr_basis, beta=args.rms_beta, optim=args.basis_optim)
                elif grid.basis_type == svox2.BASIS_TYPE_MLP:
                    optim_basis_mlp.step()
                    optim_basis_mlp.zero_grad()


    train_step()
    gc.collect()
    gstep_id_base += batches_per_epoch

    #  ckpt_path = path.join(args.train_dir, f'ckpt_{epoch_id:05d}.npz')
    # Overwrite prev checkpoints since they are very huge
    if args.save_every > 0 and (epoch_id + 1) % max(
            factor, args.save_every) == 0 and not args.tune_mode:
        print('Saving', ckpt_path)
        grid.save(ckpt_path, compress=True)

    if (gstep_id_base - last_upsamp_step) >= args.upsamp_every:
        last_upsamp_step = gstep_id_base
        if reso_id < len(reso_list) - 1:
            print('* Upsampling from', reso_list[reso_id], 'to', reso_list[reso_id + 1])
            if args.tv_early_only > 0:
                print('turning off TV regularization')
                args.lambda_tv = 0.0
                args.lambda_tv_sh = 0.0
            elif args.tv_decay != 1.0:
                args.lambda_tv *= args.tv_decay
                args.lambda_tv_sh *= args.tv_decay

            # grid.save(ckpt_path + '_before_resample', compress=True)
            reso_id += 1
            use_sparsify = True
            z_reso = reso_list[reso_id] if isinstance(reso_list[reso_id], int) else reso_list[reso_id][2]
            resample_start_time = datetime.now()
            grid.resample(reso=reso_list[reso_id],
                          sigma_thresh=args.density_thresh,
                          weight_thresh=args.weight_thresh / z_reso if use_sparsify else 0.0,
                          dilate=2,  # use_sparsify,
                          cameras=resample_cameras if args.thresh_type == 'weight' else None,
                          max_elements=args.max_grid_elements)
            resample_stop_time = datetime.now()
            secs = (resample_stop_time - resample_start_time).total_seconds()
            timings_file = open(os.path.join(args.train_dir, 'resample_time.txt'), 'a')
            timings_file.write(f"sec: {secs}\n")

            # grid.save(ckpt_path + '_after_resample', compress=True)

            if grid.use_background and reso_id <= 1:
                grid.sparsify_background(args.background_density_thresh)

            if args.upsample_density_add:
                grid.density_data.data[:] += args.upsample_density_add

        if factor > 1 and reso_id < len(reso_list) - 1:
            print('* Using higher resolution images due to large grid; new factor', factor)
            factor //= 2
            dset.gen_rays(factor=factor)
            dset.shuffle_rays()

    if gstep_id_base >= args.n_iters:
        print('* Final eval and save')

        print('Start estimate color')
        with Timing("color estimated"):
            color_estimated, color_weight, color_res = \
                grid.estimate_color_metric(c2ws, w2cs, intrinsics, images)
        print(f"Shape of the estimated color: {color_estimated.cpu().shape}")
        store_sh = grid.sh_data.data.clone()
        grid.sh_data.data[:, :] = color_estimated[:, :]
        eval_step(dset_test, "cf_test")
        grid.sh_data.data = store_sh.clone()

        if args.cf_loss_ray_frac > 0:
            eval_step(dset_test, "test")

        metric, _, _ = calc_color_metric(grid, color_res, color_weight)
        imrc = -10 * np.log10(metric.detach().cpu().numpy())
        print(f'IMRC: {imrc:.2f}')
        summary_writer.add_scalar('imrc', float(imrc), global_step=gstep_id_base)

        global_stop_time = datetime.now()
        secs = (global_stop_time - global_start_time).total_seconds()
        timings_file = open(os.path.join(args.train_dir, 'time_mins.txt'), 'a')
        timings_file.write(f"mins: {secs / 60}\n")
        if not args.tune_nosave:
            grid.save(ckpt_path, compress=True)
        break
