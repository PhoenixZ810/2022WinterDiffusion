import argparse
import os
import pdb

import nibabel as nib
from visdom import Visdom

viz = Visdom(env='diffusion')
import sys
import random

sys.path.append(".")
import numpy as np
import time
import torch as th
from PIL import Image
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.isicloader import ISICDataset
from guided_diffusion.FDSTloader import FDSTDataset
import torchvision.utils as vutils
from guided_diffusion.utils import staple
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import torchvision.transforms as transforms
from torchsummary import summary
from matplotlib import pyplot as plt
from matplotlib import cm as CM
import datetime
import traceback

seed = 10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img


def main():
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    args = create_argparser().parse_args()
    dist_util.setup_dist(args)
    logger.configure(dir=args.out_dir, arg=args)

    logger.log(time)
    if args.data_name == 'ISIC':
        tran_list = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(), ]
        transform_test = transforms.Compose(tran_list)

        ds = ISICDataset(args, args.data_dir, transform_test, mode='Test')
        args.in_ch = 4
    elif args.data_name == 'BRATS':
        tran_list = [transforms.Resize((args.image_size, args.image_size)), ]
        transform_test = transforms.Compose(tran_list)

        ds = BRATSDataset(args.data_dir, transform_test)
        args.in_ch = 5
    elif args.data_name == 'FDST':
        tran_list = [transforms.ToTensor()]
        transform_train = transforms.Compose(tran_list)
        ds = FDSTDataset(args.data_dir, 256, transform_train, test_flag=True)
        args.in_ch = 4
        args.image_size = 256
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False)
    data = iter(datal)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    all_images = []

    '''将多GPU和单GPU的模型进行统一'''
    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove `module.`
        if 'module.' in k:
            new_state_dict[k[7:]] = v
            # load params
        else:
            new_state_dict = state_dict

    model.load_state_dict(new_state_dict)
    #
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    num_image = 0
    err1 = 0
    err2 = 0
    try:
        while len(all_images) * args.batch_size < args.num_samples:
            b, m, path = next(data)  # should return an image from the dataloader "data"
            # viz.image(b[0].cpu().numpy(), opts=dict(title='images'))
            # pdb.set_trace()
            # plt.imshow(b[0].numpy().transpose(1,2,0))
            # plt.show()
            # pdb.set_trace()
            c = th.randn_like(b[:, :1, ...])
            img = th.cat((b, c), dim=1)  # add a noise channel$
            slice_ID = []
            for i in range(b.shape[0]):
                if args.data_name == 'FDST':
                    slice_ID.append(path[i].split("/")[-2] + "_" + path[i].split("/")[-1].split('.')[0])
                else:
                    slice_ID.append(path[i].split("_")[-1].split('.')[0])

            logger.log("sampling...")

            start = th.cuda.Event(enable_timing=True)
            end = th.cuda.Event(enable_timing=True)
            enslist = []

            for i in range(args.num_ensemble):  # this is for the generation of an ensemble of 5 masks.
                model_kwargs = {}
                start.record()
                sample_fn = (
                    diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
                )
                # pdb.set_trace()
                if args.data_name == 'FDST':
                    sample, x_noisy, org = sample_fn(
                        model,
                        (args.batch_size, 3, 160, 256), img,
                        step=args.diffusion_steps,
                        clip_denoised=args.clip_denoised,
                        model_kwargs=model_kwargs,
                    )
                else:
                    sample, x_noisy, org, cal, cal_out = sample_fn(
                        model,
                        (args.batch_size, 3, args.image_size, args.image_size), img,
                        step=args.diffusion_steps,
                        clip_denoised=args.clip_denoised,
                        model_kwargs=model_kwargs,
                    )

                end.record()
                th.cuda.synchronize()
                logger.log('time for 1 sample = ' + str
                (start.elapsed_time(end)) + 'ms')  # time measurement for the generation of 1 sample

                # co = th.tensor(cal_out).repeat(1, 3, 1, 1)  #
                # co = th.tensor(cal)
                # enslist.append(co)
                #
                if args.debug:
                    s = th.tensor(sample)[:, -1, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
                    o = th.tensor(org)[:, :-1, :, :]
                    c = th.tensor(cal).repeat(1, 3, 1, 1)

                    tup = (o, s, c, co)

                    compose = th.cat(tup, 0)
                    vutils.save_image(compose, fp=args.out_dir + str(slice_ID) + '_output' + str(i) + ".jpg", nrow=1,
                                      padding=10)
            # ensres = staple(th.stack(enslist,dim=0)).squeeze(0)
            # vutils.save_image(ensres, fp = args.out_dir +str(slice_ID)+'_output_ens'+".jpg", nrow = 1, padding = 10)
            plt.figure()
            save_dir = args.out_dir + 'image' + args.model_path[-9:-3]
            if not os.path.exists(save_dir):  # 判断所在目录下是否有该文件名的文件夹
                os.mkdir(save_dir)  # 创建多级目录用mkdirs，单击目录mkdir
            # print(path)
            for i in range(sample.shape[0]):
                plt.imshow(np.clip(sample[i, ...].cpu().numpy().squeeze(0), 0, 255), cmap=CM.jet)
                # plt.show()

                plt.savefig(save_dir + '/' + str(slice_ID[i]) + '_output' + ".jpg")
                plt.clf()
                # pdb.set_trace()
                '''频域'''
                fre = np.fft.fft2(np.clip(sample[i, ...].cpu().numpy().squeeze(0), 0, 255))
                fre_cen = np.fft.fftshift(fre)
                plt.imshow(np.log(1 + np.abs(fre_cen)), "gray"), plt.title("Centered Spectrum")
                plt.savefig(save_dir + '/' + str(slice_ID[i]) + '_fre' + ".jpg")
                plt.clf()
                # plt.imshow(np.clip(cal.cpu().numpy().squeeze(0).squeeze(0), 0, 255), cmap=CM.jet)
                # plt.show()
                # plt.savefig(args.out_dir + 'image/' + str(slice_ID[i]) + '_cal' + ".jpg")

                # viz.image(sample.cpu().numpy().squeeze(0), opts=dict(tcitle='den'))
                # viz.image(cal.cpu().numpy().squeeze(0), opts=dict(title='co'))
                logger.log("filename = %s" % path[i])
                num_predict = np.clip(sample[i, ...].cpu().numpy().squeeze(0), 0, 1).sum() / 10
                logger.log("sample_predict_count = " + str(num_predict
                    ) + ',' + "true_count = " + str(float(m[i])))
                err1 += abs(num_predict - float(m[i]))
                # logger.log("cal_predict_count = " + str(
                #     np.clip(cal.cpu().numpy().squeeze(0).squeeze(0), 0, 255).sum() / 5) + ',' + "true_count = " + str(float(m)))
                # err2 += abs(np.clip(cal.cpu().numpy().squeeze(0).squeeze(0), 0, 255).sum() / 5 - float(m))
                # pdb.set_trace()
                num_image += 1
                logger.log('sampleAverage_err = ' + str(err1 / num_image))
            # logger.log('calAverage_err = ' + str(err2 / num_image))
            print('sample number = %d' % int(num_image))
    except Exception as e:
        logger.log(traceback.format_exc())
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.log(time)

def create_argparser():
    defaults = dict(
        trainortest='test',
        data_name='BRATS',
        data_dir="../dataset/brats2020/testing",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
        num_ensemble=1,  # number of samples in the ensemble
        gpu_dev="0",
        out_dir='./results/',
        multi_gpu=None,  # "0,1,2"
        debug=False
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
