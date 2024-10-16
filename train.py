import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
#from util.visualizer import Visualizer
import tempfile
import os
import wandb
os.environ['WANDB_API_KEY'] = '6a4e3c8ae2f28b0b79564b0d3290599456d0c71f'

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options

    # temp_dir = tempfile.mkdtemp()
    # wandb.init(project='UNA-GAN', name=opt.name, dir=temp_dir)

    if(opt.single_channel == 1):
        opt.input_nc = 1
        opt.output_nc = 1

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.

    model = create_model(opt)      # create a model given opt.model and other options
    print('The number of training images = %d' % dataset_size)

   # visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    #opt.visualizer = visualizer
    total_iters = 0                # the total number of training iterations

    optimize_time = 0.1

    # confusion_matrix = f'/share/nas169/jethrowang/NADA-GAN/UNA-GAN/confusion_matrix/{opt.name}'
    # os.makedirs(confusion_matrix, exist_ok=True)
    # confusion_matrix_results = f'{confusion_matrix}/results.txt'
    # with open(confusion_matrix_results, 'w') as f:
        # f.write('Discriminator Training Results\n\n')

    times = []
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        # visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        # y_true = []
        # y_pred = []

        dataset.set_epoch(epoch)
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            # model.collect_D_results(y_true, y_pred)

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
               # visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(opt.name)  # it's useful to occasionally show the experiment name on console
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)            

        # model.confusion_matrix(epoch, 10, y_true, y_pred, confusion_matrix, confusion_matrix_results)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
    
    # wandb.finish()
