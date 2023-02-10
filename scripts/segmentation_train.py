
import sys
import argparse
sys.path.append("..")
sys.path.append(".")
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.isicloader import ISICDataset
from guided_diffusion.FDSTloader import FDSTDataset
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch as th
from guided_diffusion.train_util import TrainLoop
from visdom import Visdom
viz = Visdom(port=8850)
import torchvision.transforms as transforms
import pdb
# from pycallgraph import PyCallGraph
# from pycallgraph.output import GraphvizOutput
# from pycallgraph import Config
# from pycallgraph import GlobbingFilter


def main():
    args = create_argparser().parse_args()  #创建各类超参数

    dist_util.setup_dist(args)
    logger.configure(dir = args.out_dir)

    logger.log("creating data loader...")

    if args.data_name == 'ISIC':
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]  #传入需要对图片进行的操作
        transform_train = transforms.Compose(tran_list)  # 生成操作容器，便于对图片进行操作

        ds = ISICDataset(args, args.data_dir, transform_train)  # 创建dataset类
        args.in_ch = 4
    elif args.data_name == 'BRATS':
        tran_list = [transforms.Resize((args.image_size,args.image_size)),]
        transform_train = transforms.Compose(tran_list)

        ds = BRATSDataset(args.data_dir, transform_train, test_flag=False)
        args.in_ch = 5
    elif args.data_name == 'FDST':
        tran_list = [transforms.Resize((240, 135)), transforms.ToTensor(), ]
        transform_train = transforms.Compose(tran_list)

        ds = FDSTDataset(args.data_dir, transform_train, test_flag=False)
        args.in_ch = 4
        args.image_size = 256

    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True)  # 生成dataloader
    data = iter(datal)  # 生成迭代对象

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )  # 生成模型和diffusion方法
    if args.multi_gpu:
        model = th.nn.DataParallel(model,device_ids=[int(id) for id in args.multi_gpu.split(',')])
        model.to(device = th.device('cuda', int(args.gpu_dev)))
    else:
        model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=args.diffusion_steps)


    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data,
        dataloader=datal,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_name = 'BRATS',
        data_dir="../dataset/brats2020/training",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=5000,
        resume_checkpoint='',#'"./results/pretrainedmodel.pt",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpu_dev = "0",
        multi_gpu = None, #"0,1,2"
        out_dir='./results/'
    )
    defaults.update(model_and_diffusion_defaults())#将defaults训练参数和model diffusion超参数合并为同一个词典
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)#将超参词典转化为超参
    return parser


if __name__ == "__main__":
    # config = Config()
    # graphviz = GraphvizOutput()
    # graphviz.output_file = 'graph.png'
    # with PyCallGraph(output=graphviz, config=config):
    #     main()
    main()
