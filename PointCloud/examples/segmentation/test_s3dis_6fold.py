"""
Distributed training script for scene segmentation with S3DIS dataset
"""
import __init__
import argparse
import yaml
import os
import logging
import numpy as np
import glob
import pathlib
import wandb
from main import write_to_csv, test, generate_data_list
from openpoints.models import build_model_from_cfg
from openpoints.utils import get_mious, ConfusionMatrix
from openpoints.utils import set_random_seed, load_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb, generate_exp_directory, EasyConfig, dist_utils
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def test_one_room(cfg, area=5):
    logging.info(f'================ Area {area} ================')
    model = build_model_from_cfg(cfg.model).to(cfg.rank)
    model_size = cal_model_parm_nums(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))

    best_epoch, best_val = load_checkpoint(
        model, pretrained_path=cfg.pretrained_path)
    cfg.dataset.common.test_area = area
    data_list = generate_data_list(cfg)
    logging.info(f"length of test dataset: {len(data_list)}")
    test_miou, test_macc, test_oa, test_ious, test_accs, all_cm = test(
        model, data_list, cfg, num_votes=1)
    cfg.allarea_cm.value += all_cm.value
    with np.printoptions(precision=2, suppress=True):
        logging.info(
            f'Best ckpt @E{best_epoch},  test_oa {test_oa:.2f}, test_macc {test_macc:.2f}, test_miou {test_miou:.2f}, '
            f'\niou per cls is: {test_ious}')
    write_to_csv(test_oa, test_macc, test_miou, test_ious,
                 best_epoch, cfg, write_header=area == 1, area=area)
    logging.info(f'save results in {cfg.csv_path}')
    logging.info(f'================ End of Area {area} ================\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser('S3DIS scene segmentation training')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--profile', action='store_true',
                        default=False, help='set to True to profile speed')
    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)

    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)
    # init distributed env first, since logger depends on the dist info.
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(
        cfg)

    cfg.mode = 'test'
    # init log dir
    cfg.task_name = args.cfg.split('.')[-2].split('/')[-2]
    cfg.cfg_basename = args.cfg.split('.')[-2].split('/')[-1]  # cfg_basename, \eg pointnext-xl
    tags = [
        cfg.task_name,  # task name (the folder of name under ./cfgs
        cfg.mode,
        cfg.cfg_basename,  # cfg file name
        f'ngpus{cfg.world_size}',
        f'seed{cfg.seed}',
    ]
    cfg.exp_name = args.cfg.split('.')[-2].split('/')[-1]
    tags = [
        cfg.task_name,  # task name (the folder of name under ./cfgs
        cfg.mode,
        'all_areas',
        cfg.exp_name,  # cfg file name
    ]
    for i, opt in enumerate(opts):
        if 'rank' not in opt and 'dir' not in opt and 'root' not in opt and 'path' not in opt and 'wandb' not in opt and '/' not in opt:
            tags.append(opt)
    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)

    cfg.is_training = cfg.mode in [
        'train', 'training', 'finetune', 'finetuning']

    generate_exp_directory(
        cfg, tags, additional_id=os.environ.get('MASTER_PORT', None))
    cfg.wandb.tags = tags

    os.environ["JOB_LOG_DIR"] = cfg.log_dir
    cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, indent=2)
        os.system('cp %s %s' % (args.cfg, cfg.run_dir))
    cfg.cfg_path = cfg_path

    # wandb config
    cfg.wandb.name = cfg.run_name
    # logger
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    if cfg.rank == 0:
        Wandb.launch(cfg, cfg.wandb.use_wandb)
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)

    cfg.csv_path = os.path.join(cfg.run_dir, cfg.run_name + f'_allareas.csv')
    # Get the path of each pretrained
    pretrained_paths_unorder = glob.glob(
        str(pathlib.Path(cfg.pretrained_path) / '*' / '*checkpoint*' / '*_best.pth'))
    cfg.allarea_cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)

    cfg.classes = ['ceiling',
                   'floor',
                   'wall',
                   'beam',
                   'column',
                   'window',
                   'door',
                   'chair',
                   'table',
                   'bookcase',
                   'sofa',
                   'board',
                   'clutter']

    for i in range(1, 7):  # 6 areas
        cfg.dataset.common.test_area = i

        # find the corret pretrained path from the list
        pretrained_path = None

        for pretrained in pretrained_paths_unorder:
            if f'test_area={i}' in pretrained:
                pretrained_path = pretrained
                break
            if not 'test_area' in pretrained and i == 5:
                pretrained_path = pretrained
                logging.info(f'assume {pretrained} is on Area 5')
                break
        assert pretrained_path is not None, f'fail to find pretrained_path for area {i}'
        cfg.pretrained_path = pretrained_path
        test_one_room(cfg, area=i)

    # all area
    tp, union, count = cfg.allarea_cm.tp, cfg.allarea_cm.union, cfg.allarea_cm.count
    logging.info(f'the total number of points of all areas are: {count}')
    miou, macc, oa, ious, accs = get_mious(tp, union, count)
    with np.printoptions(precision=2, suppress=True):
        logging.info(f'test_oa {oa:.2f}, test_macc {macc:.2f}, test_miou {miou:.2f}, '
                     f'\niou per cls is: {ious}')
    write_to_csv(oa, macc, miou, ious, '-', cfg,
                 write_header=False, area='all')
