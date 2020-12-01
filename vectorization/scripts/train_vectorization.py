#!/usr/bin/env python3

import argparse
from collections import defaultdict
from itertools import islice
import os.path
import sys

import numpy as np
import torch
import torch.nn
import torch.optim

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..')
)
sys.path[1:1] = [__dir__]

from util_files.logging import create_logger
import util_files.dataloading as dataloading
from vectorization import load_model
from util_files.os import require_empty
from util_files.visualization import make_ranked_images_from_loader_and_model
from util_files.data.graphics_primitives import PT_QBEZIER
from util_files.optimization.optimizer.scheduled_optimizer import ScheduledOptimizer
from util_files.tensorboard import SummaryWriter
import util_files.metrics.vector_metrics as vmetrics
import util_files.loss_functions.supervised as supervised_loss
import os
from time import time


def prepare_batch_on_device(batch_data, device):
    images, y_true = batch_data
    images = images.to(device)
    y_true = y_true.to(device)
    return images, y_true


def main(options):
    torch.autograd.set_detect_anomaly(False)

    name, ext = os.path.splitext(os.path.basename(options.model_spec_filename))
    logs_dir = os.path.join(options.logs_dir, name)
    require_empty(logs_dir, recreate=options.overwrite)

    if None is options.logging_filename:
        options.logging_filename = os.path.join(logs_dir, 'train.log')
    if None is options.tboard_json_logging_file:
        options.tboard_json_logging_file = os.path.join(logs_dir, 'tboard.json')
    if None is options.tboard_dir:
        options.tboard_dir = logs_dir
    if None is options.save_model_filename:
        options.save_model_filename = os.path.join(logs_dir, 'model')

    #####################################################
    # all these parameters stay unchanged for all models

    max_lines = 10

    # If  l2_weight_change >= 0, options.l2_weight_init = 1 this is MSE or mapping_l2
    # If  l2_weight_change <= 0, options.l2_weight_init = 0 this is L1

    l2_weight_change = options.l2_weight_change
    l2_weight = options.l2_weight_init
    parallel = False
    print(options.gpu)
    if len(options.gpu) == 0:
        device = torch.device('cpu')
        prefetch_data = False
    elif len(options.gpu) == 1:
        device = torch.device('cuda:{}'.format(options.gpu[0]))
        prefetch_data = True
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(options.gpu)
        device = torch.device('cuda:{}'.format(options.gpu[0]))
        prefetch_data = True
        parallel = True
    RASTER_RES = (options.render_res, options.render_res)

    METRIC_PARAMS_AVG = {'average': 'mean', 'binarization': 'median', 'raster_res': RASTER_RES}
    METRIC_PARAMS = {'binarization': 'median', 'raster_res': RASTER_RES}
    IMGLOG_PARAMS = {'imgrid_shape': (2, 12), 'patch_size': (150, 150), 'with_skeleton': True,
                     'stack_grids_horizontally': False, 'skeleton_node_size': 5}

    # all these parameters stay unchanged for all models
    #####################################################
    logger = create_logger(options)
    writer = SummaryWriter(logdir=options.tboard_dir)

    logger.info("Called with parameters: {}".format(options.__dict__))

    make_loaders_fn = dataloading.prepare_loaders[options.dataloader_type]
    train_loader_memory_constraint = options.memory_constraint  # TODO could be calculated more precisely with respect to the other memory requirements
    loader_params = dict(data_root=options.data_root, train_batch_size=options.train_batch_size,
                         val_batch_size=options.val_batch_size,
                         mini_val_batches_n_per_subset=options.mini_val_batches_n_per_subset,
                         memory_constraint=train_loader_memory_constraint,
                         shuffle_train=True, prefetch=prefetch_data, device=device)
    if options.dataloader_type == 'handcrafted':
        loader_params.update(handcrafted_train_paths=options.handcrafted_train_paths,
                             handcrafted_val_paths=options.handcrafted_val_paths,
                             handcrafted_val_part=options.handcrafted_val_part)
    train_loader, val_loader, val_mini_loader = make_loaders_fn(**loader_params)
    logger.info('Total number of train patches: ~{}'.format(len(train_loader) * options.train_batch_size))
    logger.info('Total number of val patches: ~{}'.format(len(val_loader) * options.val_batch_size))
    logger.info('Total number of mini val patches: ~{}'.format(len(val_mini_loader) * options.val_batch_size))

    model = load_model(options.model_spec_filename).to(device)
    logger.info_trainable_params(model)

    optimizer = ScheduledOptimizer(
        torch.optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        1000, 4000)

    if options.init_model_filename:
        checkpoint = torch.load(options.init_model_filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epochs_completed = checkpoint['epoch']
        batches_completed_in_epoch = checkpoint['batch']
        optimizer.n_current_steps = checkpoint['iter']
    else:
        epochs_completed = 0
        batches_completed_in_epoch = 0

    # Loss function choose
    if parallel:
        print('parallel')
        print(torch.cuda.device_count())
        model = torch.nn.DataParallel(model)
    make_loss_fn = supervised_loss.prepare_losses[options.loss_funct]

    pt = PT_QBEZIER

    def validate(loader, log_i, prefix=''):
        val_metrics = defaultdict(list)
        val_loss = []
        # Model to eval
        model.eval()
        for batch_j, batch_data_val in enumerate(loader):
            logger.info('Validating batch {}'.format(batch_j))
            # Run through validation dataset
            with logger.print_duration('    preparing batch on device'):
                images, y_true = prepare_batch_on_device(batch_data_val, device)

            is_nan_idx = torch.isnan(images).any(-1).any(-1).any(-1)
            images[is_nan_idx] = 0
            y_true[is_nan_idx] = 0

            with torch.no_grad():
                with logger.print_duration('    forward pass'):
                    y_pred = model.forward(images, max_lines)

            val_losses_per_sample = make_loss_fn(y_pred, y_true, reduction='none').cpu().numpy()
            val_loss.extend(val_losses_per_sample)

            y_true_vector = vmetrics.batch_numpy_to_vector(y_true.cpu().numpy(), RASTER_RES, pt)
            y_pred_vector = vmetrics.batch_numpy_to_vector(y_pred.cpu().numpy(), RASTER_RES, pt)
            for metric_name, metric in vmetrics.METRICS_BY_NAME.items():
                metric_value = metric(y_true_vector, y_pred_vector, **METRIC_PARAMS)
                metric_name = '{}val_{}'.format(prefix, metric_name)
                val_metrics[metric_name].append(np.atleast_1d(metric_value))

        # Save stuff to tensorboard for visualization
        val_loss = np.asarray(val_loss)

        class _Loader:
            def __iter__(self):
                for batch_data_val in loader:
                    yield prepare_batch_on_device(batch_data_val, device)

        _loader = _Loader()

        # scores = -val_loss
        scores = np.concatenate(val_metrics['{}val_{}'.format(prefix, 'iou_score')])

        # TODO visualizatins problem
        worst_grid, best_grid, average_grid = make_ranked_images_from_loader_and_model(
            lambda input: model(input, max_lines), _loader, scores, **IMGLOG_PARAMS)
        writer.add_image('worst/{}val'.format(prefix), worst_grid, log_i)
        writer.add_image('best/{}val'.format(prefix), best_grid, log_i)
        writer.add_image('average/{}val'.format(prefix), average_grid, log_i)

        mean_val_loss = val_loss.mean()
        writer.add_scalar(('{}val_' + options.loss_funct).format(prefix), mean_val_loss, global_step=log_i)
        logger.info('Validation loss: {:.4f}'.format(mean_val_loss))

        metrics_scalars = {name: np.mean(np.concatenate(values)) for name, values in val_metrics.items()}
        for name, value in metrics_scalars.items():
            writer.add_scalar(name, value, global_step=log_i)
            values_array = np.concatenate(val_metrics[name])
            values_array[np.isnan(values_array)] = 0.  # maybe tensorboard does handle this by itself, dunno
            values_array[np.isinf(values_array)] = 0.  # otherwise error "The histogram is empty"
            if values_array.size > 0:
                writer.add_histogram(name + '_hist', values_array, global_step=log_i)
        logger.info_scalars('Computed {key} over {num_items} images: {value:.4f}', metrics_scalars,
                            num_items=len(val_loss))

    # TODO @mvkolos: add training on multiple GPUs (4x batches / sec.)
    for epoch_i in range(epochs_completed, epochs_completed + options.epochs):
        for batch_i, batch_data in islice(enumerate(train_loader), batches_completed_in_epoch, len(train_loader)):
            start_time = time()
            model.train()
            iter_i = epoch_i * len(train_loader) + batch_i
            # Train for one batch
            logger.info('Training batch {}'.format(batch_i))
            with logger.print_duration('    preparing batch on device'):
                images, y_true = prepare_batch_on_device(batch_data, device)

            is_nan_idx = torch.isnan(images).any(-1).any(-1).any(-1)
            images[is_nan_idx] = 0
            y_true[is_nan_idx] = 0

            with logger.print_duration('    forward pass'):
                y_pred = model.forward(images, max_lines)

            loss = make_loss_fn(y_pred, y_true, l2_weight=l2_weight)

            l2_weight = min(max(0., l2_weight + l2_weight_change), 1.)

            optimizer.zero_grad()

            with logger.print_duration('    backward pass'):
                loss.backward()
                optimizer.step_and_update_lr()

            # Output loss for each training step, as it is already computed
            logger.info(
                'Training iteration [{item_idx} / {num_batches}] {percent:.3f}% complete, loss: {loss:.4f}'.format(
                    item_idx=batch_i, num_batches=len(train_loader),
                    percent=100. * batch_i / len(train_loader), loss=loss.item()))
            logger.info('1 batch time  {}'.format(time() - start_time))
            writer.add_scalar('learning_rate', optimizer.get_lr()[0], global_step=iter_i)
            writer.add_scalar(('train_' + options.loss_funct), loss.item(), global_step=iter_i)

            if batch_i > 0 and batch_i % options.batches_before_val == 0:
                # Save stuff to tensorboard for visualization -- this is really expensive if done each batch
                logger.debug('    computing metrics on last train batch and logging to text files and tensorboard')
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), global_step=iter_i)
                y_true_vector = vmetrics.batch_numpy_to_vector(y_true.cpu().numpy(), RASTER_RES, pt)
                y_pred_vector = vmetrics.batch_numpy_to_vector(y_pred.detach().cpu().numpy(), RASTER_RES, pt)
                train_metrics = {'train_{}'.format(name): metric(y_true_vector, y_pred_vector, **METRIC_PARAMS_AVG)
                                 for name, metric in vmetrics.METRICS_BY_NAME.items()}
                for name, value in train_metrics.items():
                    writer.add_scalar(name, value, global_step=iter_i)
                logger.info_scalars('Computed {key} over last batch of {num_items} images: {value:.4f}', train_metrics,
                                    num_items=len(y_true))

                logger.info('Running mini validation with {} batches'.format(len(val_mini_loader)))
                validate(val_mini_loader, iter_i, prefix='mini')

            if batch_i % options.batches_before_save == 0:
                weights_filename = '{prefix}_{batch_id}.weights'.format(
                    prefix=options.save_model_filename, batch_id=iter_i)
                torch.save({
                    'epoch': epoch_i,
                    'batch': batch_i,
                    'iter': iter_i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, weights_filename)
                logger.debug('Model for batch {} saved to {}'.format(batch_i, weights_filename))

        batches_completed_in_epoch = 0

        logger.info('Running validation on the whole validation dataset with {} batches'.format(len(val_loader)))
        validate(val_loader, (epoch_i + 1) * len(train_loader))

    if options.tboard_json_logging_file:
        writer.export_scalars_to_json(options.tboard_json_logging_file)
    writer.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', action='append', help='GPU to use, can use multiple [default: use CPU].')

    parser.add_argument('-e', '--epochs', type=int, default=1, help='how many epochs to train [default: 1].')
    parser.add_argument('-b', '--train-batch-size', type=int, default=128, dest='train_batch_size',
                        help='train batch size [default: 128].')
    parser.add_argument('-B', '--val-batch-size', type=int, default=128, dest='val_batch_size',
                        help='val batch size [default: 128].')
    parser.add_argument('--batches-before-val', type=int, default=1024, dest='batches_before_val',
                        help='how many batches to train before validation [default: 1024].')
    parser.add_argument('--mini-val-batches-n-per-subset', type=int, default=12, dest='mini_val_batches_n_per_subset',
                        help='how many batches per subset to run for mini validation [default: 12].')

    parser.add_argument('--model-spec', dest='model_spec_filename', required=True,
                        help='model specification JSON file to use [default: none].')
    parser.add_argument('--infer-from-spec', dest='infer_from_spec', action='store_true', default=False,
                        help='if set, --model, --save-model-file, --logging-file, --tboard-json-logging-file,'
                             'and --tboard-dir are formed automatically [default: False].')
    parser.add_argument('--log-dir-prefix', dest='logs_dir', default='/logs',
                        help='path to root of logging location [default: /logs].')
    parser.add_argument('-m', '--init-model-file', dest='init_model_filename',
                        help='Path to initializer model file [default: none].')

    parser.add_argument('-s', '--save-model-file', dest='save_model_filename',
                        help='Path to output vectorization model file [default: none].')
    parser.add_argument('--batches_before_save', type=int, default=1024, dest='batches_before_save',
                        help='how many batches to run before saving the model [default: 1024].')

    parser.add_argument('--data-root', required=True, dest='data_root', help='root of the data tree (directory).')
    parser.add_argument('--data-type', required=True, dest='dataloader_type',
                        help='type of the train/val data to use.', choices=dataloading.prepare_loaders.keys())
    parser.add_argument('--handcrafted-train', required=False, action='append',
                        dest='handcrafted_train_paths', help='dirnames of handcrafted datasets used for training '
                                                             '(sought for in preprocessed/synthetic_handcrafted).')
    parser.add_argument('--handcrafted-val', required=False, action='append',
                        dest='handcrafted_val_paths', help='dirnames of handcrafted datasets used for validation '
                                                           '(sought for in preprocessed/synthetic_handcrafted).')
    parser.add_argument('--handcrafted-val-part', required=False, type=float, default=.1,
                        dest='handcrafted_val_part', help='portion of handcrafted_train used for validation')
    parser.add_argument('-M', '--memory-constraint', required=True, type=int, dest='memory_constraint',
                        help='maximum RAM usage in bytes.')

    parser.add_argument('-r', '--render-resolution', dest='render_res', default=64, type=int,
                        help='resolution used for rendering.')

    parser.add_argument('--loss-funct', required=False, dest='loss_funct',
                        help='Choose loss function. Default vectran_loss',
                        choices=supervised_loss.prepare_losses.keys(),
                        default='vectran_loss')

    parser.add_argument('--l2_weight_change', required=False, dest='l2_weight_change',
                        help='Weight change for L2. l2_weight = l2_weight + l2_weight_change(default -1e-5)',
                        type=float, default=-1e-5)

    parser.add_argument('--l2_weight_init', required=False, dest='l2_weight_init',
                        help='l2_weight initialization(default 1)',
                        type=float, default=1.)

    parser.add_argument('--verbose', action='store_true', default=False, dest='verbose',
                        help='verbose output [default: False].')
    parser.add_argument('-l', '--logging-file', dest='logging_filename',
                        help='Path to output logging text file [default: output to stdout only].')
    parser.add_argument('-tl', '--tboard-json-logging-file', dest='tboard_json_logging_file',
                        help='Path to output logging JSON file with scalars [default: none].')
    parser.add_argument('-x', '--tboard-dir', dest='tboard_dir',
                        help='Path to tensorboard [default: do not log events].')
    parser.add_argument('-w', '--overwrite', action='store_true', default=False,
                        help='If set, overwrite existing logs [default: exit if output dir exists].')

    # add two more args corresponding to the parsing of job args
    parser.add_argument('-jac', '--job-array-config', dest='job_array_config',
                        help='Job array-formatted file (one set of commandline parameters per file).')
    parser.add_argument('-jati', '--job-array-task-id', dest='job_array_task_id', type=int,
                        help='Task ID to parse from the job array-formatted file.')

    # parse the actual commandline
    options = parser.parse_args()

    if None is not options.job_array_config and None is not options.job_array_config:
        # get a parser
        for action in parser._actions:
            action.required = False  # so that we won't get errors when the action
            action.default = None  # so that we won't get falsely set args when they're not really encountered

        # parse the job array command line
        with open(options.job_array_config) as jac_file:
            lines = [line.strip() for line in jac_file.readlines()]
            job_array_options = parser.parse_args(lines[options.job_array_task_id].split())

        for name, value in job_array_options.__dict__.items():
            if None is not value:
                setattr(options, name, value)

    return options


if __name__ == '__main__':
    options = parse_args()
    main(options)
