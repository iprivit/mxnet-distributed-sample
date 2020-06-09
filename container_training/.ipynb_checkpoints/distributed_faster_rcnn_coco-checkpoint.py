import argparse
import logging
import os
import tarfile
import time

import mxnet as mx
import horovod.mxnet as hvd
from mxnet import autograd, gluon, nd
from mxnet.test_utils import download
import gluoncv
from gluoncv import model_zoo
from gluoncv.utils import  makedirs # ,download
from mxnet.test_utils import download
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
import horovod.mxnet as hvd
from gluoncv import data as gdata
from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.data.batchify import FasterRCNNTrainBatchify, Tuple, Append
from gluoncv.data.transforms.presets.rcnn import FasterRCNNDefaultTrainTransform, \
    FasterRCNNDefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.parallel import Parallel, Parallelizable
from gluoncv.utils.metrics.rcnn import RPNAccMetric, RPNL1LossMetric, RCNNAccMetric, \
    RCNNL1LossMetric
from gluoncv.model_zoo.rcnn.faster_rcnn.data_parallel import ForwardBackwardTask
import boto3

def get_lr_at_iter(alpha, lr_warmup_factor=1. / 3.):
    return lr_warmup_factor * (1 - alpha) + alpha

    
def split_and_load(batch, ctx_list):
    """Split data to 1 batch each device."""
    new_batch = []
    for i, data in enumerate(batch):
        if isinstance(data, (list, tuple)):
            new_data = [x.as_in_context(ctx) for x, ctx in zip(data, ctx_list)]
        else:
            new_data = [data.as_in_context(ctx_list[0])]
        new_batch.append(new_data)
    return new_batch

def download_voc(path, overwrite=False):
    _DOWNLOAD_URLS = [
        ('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
         '34ed68851bce2a36e2a223fa52c661d592c66b3c'),
        ('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
         '41a8d6e12baa5ab18ee7f8f8029b9e11805b4ef1'),
        ('http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
         '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')]
    makedirs(path)
    for url, checksum in _DOWNLOAD_URLS:
#         try:
#             filename = download(url, path=path, overwrite=overwrite, sha1_hash=checksum)
#         except:
        filename = download(url, dirname=path)
        # extract
        with tarfile.open(filename) as tar:
            tar.extractall(path=path)

def main():

        
    # Function to get mnist iterator given a rank
    def get_voc_iterator(rank, num_workers, net, num_shards):
        data_dir = "data-%d" % rank
        try:
            s3_client = boto3.client('s3')
            for file in ['VOCtrainval_06-Nov-2007.tar','VOCtest_06-Nov-2007.tar','VOCtrainval_11-May-2012.tar']:
                s3_client.download_file(args.s3bucket,f'voc_tars/{file}',f'/opt/ml/code/{file}')
                with tarfile.open(filename) as tar:
                    tar.extractall(path=path)
        except:
            print('downloading from source')
            download_voc(data_dir)

        input_shape = (1, 256, 256, 3)
        batch_size = args.batch_size
        
        # might want to replace with mx.io.ImageDetRecordIter, this means you need data in RecordIO format 
#         train_iter = mx.io.MNISTIter(
#             image="%s/train-images-idx3-ubyte" % data_dir,
#             label="%s/train-labels-idx1-ubyte" % data_dir,
#             input_shape=input_shape,
#             batch_size=batch_size,
#             shuffle=True,
#             flat=False,
#             num_parts=hvd.size(),
#             part_index=hvd.rank()
#         )
        
        train_dataset = gdata.VOCDetection(root=f'/opt/ml/code/data-{rank}/VOCdevkit/',
            splits=[(2007, 'trainval'), (2012, 'trainval')])
        val_dataset = gdata.VOCDetection(root=f'/opt/ml/code/data-{rank}/VOCdevkit/',
            splits=[(2007, 'test')])
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
        im_aspect_ratio = [1.] * len(train_dataset)
        train_bfn = FasterRCNNTrainBatchify(net)
        train_sampler = gluoncv.nn.sampler.SplitSortedBucketSampler(im_aspect_ratio, batch_size,
                                                        num_parts=hvd.size() if args.horovod else 1,
                                                        part_index=hvd.rank() if args.horovod else 0,
                                                        shuffle=True)
        # had issue with multi_stage=True
        train_iter = mx.gluon.data.DataLoader(train_dataset.transform(
        FasterRCNNDefaultTrainTransform(net.short, net.max_size, net, ashape=net.ashape, multi_stage=False)),
        batch_sampler=train_sampler, batchify_fn=train_bfn, num_workers=num_workers)
        
        val_bfn = Tuple(*[Append() for _ in range(3)])
        short = net.short[-1] if isinstance(net.short, (tuple, list)) else net.short
        # validation use 1 sample per device
        val_iter = mx.gluon.data.DataLoader(
            val_dataset.transform(FasterRCNNDefaultValTransform(short, net.max_size)), num_shards, False,
            batchify_fn=val_bfn, last_batch='keep', num_workers=num_workers)

        return train_iter, val_iter


    # Function to define neural network
    def conv_nets(model_name):
        net = model_zoo.get_model(model_name, pretrained_base=False)
        return net


    def evaluate(net, val_data, ctx, eval_metric, args):
        """Test on validation dataset."""
        clipper = gcv.nn.bbox.BBoxClipToImage()
        eval_metric.reset()
        if not args.disable_hybridization:
            # input format is differnet than training, thus rehybridization is needed.
            net.hybridize(static_alloc=args.static_alloc)
        for batch in val_data:
            batch = split_and_load(batch, ctx_list=ctx)
            det_bboxes = []
            det_ids = []
            det_scores = []
            gt_bboxes = []
            gt_ids = []
            gt_difficults = []
            for x, y, im_scale in zip(*batch):
                # get prediction results
                ids, scores, bboxes = net(x)
                det_ids.append(ids)
                det_scores.append(scores)
                # clip to image size
                det_bboxes.append(clipper(bboxes, x))
                # rescale to original resolution
                im_scale = im_scale.reshape((-1)).asscalar()
                det_bboxes[-1] *= im_scale
                # split ground truths
                gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
                gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
                gt_bboxes[-1] *= im_scale
                gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

            # update metric
            for det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff in zip(det_bboxes, det_ids,
                                                                            det_scores, gt_bboxes,
                                                                            gt_ids, gt_difficults):
                eval_metric.update(det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff)
        return eval_metric.get()


    # Initialize Horovod
    hvd.init()

    # Horovod: pin context to local rank
    if args.horovod:
        ctx = [mx.gpu(hvd.local_rank())]
    else:
        ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
        ctx = ctx if ctx else [mx.cpu()]
    context = mx.cpu(hvd.local_rank()) if args.no_cuda else mx.gpu(hvd.local_rank())
    num_workers = hvd.size()


    # Build model
    model = conv_nets(args.model_name)
    model.cast(args.dtype)
    model.hybridize()
    
    # Initialize parameters
    initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in",
                                 magnitude=2)
    model.initialize(initializer, ctx=context)
    
    # Create optimizer
    optimizer_params = {'momentum': args.momentum,
                        'learning_rate': args.lr * hvd.size()}
    opt = mx.optimizer.create('sgd', **optimizer_params)
    
    # Load training and validation data
    train_data, val_data = get_voc_iterator(hvd.rank(),num_workers, model, len(ctx))

    # Horovod: fetch and broadcast parameters
    params = model.collect_params()
    if params is not None:
        hvd.broadcast_parameters(params, root_rank=0)

    # Horovod: create DistributedTrainer, a subclass of gluon.Trainer
    trainer = hvd.DistributedTrainer(params, opt)

    # Create loss function and train metric
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
    # adding in new loss functions 
    rpn_cls_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
    rpn_box_loss = mx.gluon.loss.HuberLoss(rho=args.rpn_smoothl1_rho)  # == smoothl1
    rcnn_cls_loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    rcnn_box_loss = mx.gluon.loss.HuberLoss(rho=args.rcnn_smoothl1_rho)  # == smoothl1
    metrics = [mx.metric.Loss('RPN_Conf'),
               mx.metric.Loss('RPN_SmoothL1'),
               mx.metric.Loss('RCNN_CrossEntropy'),
               mx.metric.Loss('RCNN_SmoothL1'), ]

    rpn_acc_metric = RPNAccMetric()
    rpn_bbox_metric = RPNL1LossMetric()
    rcnn_acc_metric = RCNNAccMetric()
    rcnn_bbox_metric = RCNNL1LossMetric()
    metrics2 = [rpn_acc_metric, rpn_bbox_metric, rcnn_acc_metric, rcnn_bbox_metric]
    
    metric = mx.metric.Accuracy()


    # Global training timing
    if hvd.rank()==0:
        global_tic = time.time()

    # Train model
#     for epoch in range(args.epochs):
#         tic = time.time()
#         train_data.reset()
#         metric.reset()
#         for nbatch, batch in enumerate(train_data, start=1):
#             data = batch.data[0].as_in_context(context)
#             label = batch.label[0].as_in_context(context)
#             with autograd.record():
#                 output = model(data.astype(args.dtype, copy=False))
#                 loss = loss_fn(output, label)
#             loss.backward()
#             trainer.step(args.batch_size)
#             metric.update([label], [output])

#             if nbatch % 100 == 0:
#                 name, acc = metric.get()
#                 logging.info('[Epoch %d Batch %d] Training: %s=%f' %
#                              (epoch, nbatch, name, acc))

#         if hvd.rank() == 0:
#             elapsed = time.time() - tic
#             speed = nbatch * args.batch_size * hvd.size() / elapsed
#             logging.info('Epoch[%d]\tSpeed=%.2f samples/s\tTime cost=%f',
#                          epoch, speed, elapsed)

#         # Evaluate model accuracy
#         _, train_acc = metric.get()
#         name, val_acc = evaluate(model, val_data, context)
#         if hvd.rank() == 0:
#             logging.info('Epoch[%d]\tTrain: %s=%f\tValidation: %s=%f', epoch, name,
#                          train_acc, name, val_acc)
        

#     if hvd.rank()==0:
#         global_training_time =time.time() - global_tic
#         print("Global elpased time on training:{}".format(global_training_time))
#         device = context.device_type + str(num_workers)
        
    # train from train_faster_rcnn.py 
    for epoch in range(args.epochs):
        lr_decay = float(args.lr_decay)
        lr_steps = sorted([float(ls) for ls in args.lr_decay_epoch.split(',') if ls.strip()])
        lr_warmup = float(args.lr_warmup)  # avoid int division
        # this simplifies dealing with all of the loss functions 
        rcnn_task = ForwardBackwardTask(model, trainer, rpn_cls_loss, rpn_box_loss, rcnn_cls_loss,
                                        rcnn_box_loss, mix_ratio=1.0, amp_enabled=args.amp)
        executor = Parallel(args.executor_threads, rcnn_task) if not args.horovod else None
        mix_ratio = 1.0
        if not args.disable_hybridization:
            model.hybridize(static_alloc=args.static_alloc)
        if args.mixup:
            # TODO(zhreshold) only support evenly mixup now, target generator needs to be modified otherwise
            train_data._dataset._data.set_mixup(np.random.uniform, 0.5, 0.5)
            mix_ratio = 0.5
            if epoch >= args.epochs - args.no_mixup_epochs:
                train_data._dataset._data.set_mixup(None)
                mix_ratio = 1.0
        while lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer.learning_rate * lr_decay
            lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)
            logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))
        for metric in metrics:
            metric.reset()
        tic = time.time()
        btic = time.time()
        base_lr = trainer.learning_rate
        rcnn_task.mix_ratio = mix_ratio
        for i, batch in enumerate(train_data):
            if epoch == 0 and i <= lr_warmup: # does a learning rate reset if warming up 
                # adjust based on real percentage
                new_lr = base_lr * get_lr_at_iter(i / lr_warmup, args.lr_warmup_factor)
                if new_lr != trainer.learning_rate:
                    if i % args.log_interval == 0:
                        logger.info(
                            '[Epoch 0 Iteration {}] Set learning rate to {}'.format(i, new_lr))
                    trainer.set_learning_rate(new_lr)
            batch = split_and_load(batch, ctx_list=ctx) # does split and load function, creates a batch per device 
            metric_losses = [[] for _ in metrics]
            add_losses = [[] for _ in metrics2]
            if executor is not None:
                for data in zip(*batch):
                    executor.put(data)
            for j in range(len(ctx)):
                if executor is not None:
                    result = executor.get()
                else:
                    result = rcnn_task.forward_backward(list(zip(*batch))[0])
                if (not args.horovod) or hvd.rank() == 0:
                    for k in range(len(metric_losses)):
                        metric_losses[k].append(result[k])
                    for k in range(len(add_losses)):
                        add_losses[k].append(result[len(metric_losses) + k])
            for metric, record in zip(metrics, metric_losses):
                metric.update(0, record)
            for metric, records in zip(metrics2, add_losses):
                for pred in records:
                    metric.update(pred[0], pred[1])
            trainer.step(batch_size)

            # update metrics
            if (not args.horovod or hvd.rank() == 0) and args.log_interval \
                    and not (i + 1) % args.log_interval:
                msg = ','.join(
                    ['{}={:.3f}'.format(*metric.get()) for metric in metrics + metrics2])
                logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}'.format(
                    epoch, i, args.log_interval * args.batch_size / (time.time() - btic), msg))
                btic = time.time()

        if (not args.horovod) or hvd.rank() == 0:
            msg = ','.join(['{}={:.3f}'.format(*metric.get()) for metric in metrics])
            logger.info('[Epoch {}] Training cost: {:.3f}, {}'.format(
                epoch, (time.time() - tic), msg))
            if not (epoch + 1) % args.val_interval:
                # consider reduce the frequency of validation to save time
                map_name, mean_ap = validate(model, val_data, ctx, eval_metric, args)
                val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
                logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
                current_map = float(mean_ap[-1])
            else:
                current_map = 0.
            save_params(model, logger, best_map, current_map, epoch, args.save_interval,
                        args.save_prefix)


if __name__ == "__main__":
    # Handling script arguments
    parser = argparse.ArgumentParser(description='MXNet PASCAL VOC Distributed Example')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size (default: 32)')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='training data type (default: float32)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of training epochs (default: 5)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-epoch', type=str, default='',
                        help='epochs at which learning rate decays. default is 14,20 for voc.')
    parser.add_argument('--lr-warmup', type=str, default='0',
                        help='warmup iterations to adjust learning rate, default is 0 for voc.')
    parser.add_argument('--lr-warmup-factor', type=float, default=1. / 3.,
                        help='warmup factor of base lr.')
    parser.add_argument('--wd', type=str, default='',
                        help='Weight decay, default is 5e-4 for voc')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--disable-hybridization', action='store_true',
                        help='Whether to disable hybridize the model. '
                             'Memory usage and speed will decrese.')
    parser.add_argument('--executor-threads', type=int, default=1,
                        help='Number of threads for executor for scheduling ops. '
                             'More threads may incur higher GPU memory footprint, '
                             'but may speed up throughput. Note that when horovod is used, '
                             'it is set to 1.')
    parser.add_argument('--horovod', action='store_true',default=True,
                        help='Use MXNet Horovod for distributed training. Must be run with OpenMPI. '
                             '--gpus is ignored when using --horovod.')
    parser.add_argument('--amp', action='store_true',
                        help='Use MXNet AMP for mixed precision training.')
    parser.add_argument('--static-alloc', action='store_true',
                        help='Whether to use static memory allocation. Memory usage will increase.')
    parser.add_argument('--mixup', action='store_true', help='Use mixup training.')
    parser.add_argument('--no-mixup-epochs', type=int, default=20,
                        help='Disable mixup training if enabled in the last N epochs.')
    parser.add_argument('--no-cuda', action='store_true', help='disable training on GPU (default: False)')
    parser.add_argument('--rpn-smoothl1-rho', type=float, default=1. / 9.,
                        help='RPN box regression transition point from L1 to L2 loss.'
                             'Set to 0.0 to make the loss simply L1.')
    parser.add_argument('--rcnn-smoothl1-rho', type=float, default=1.,
                        help='RCNN box regression transition point from L1 to L2 loss.'
                             'Set to 0.0 to make the loss simply L1.')
    parser.add_argument('--model-name', type=str, default='faster_rcnn_resnet101_v1d_coco')
    parser.add_argument('--s3bucket', type=str, default='privisaa-bucket-virginia') # CHANGE THIS
    args = parser.parse_args()

    if not args.no_cuda:
        # Disable CUDA if there are no GPUs.
        if mx.context.num_gpus() == 0:
            args.no_cuda = True

    logging.basicConfig(level=logging.INFO)
    logging.info(args)
   
    main()
