import os
import time
import torch


def train_one_epoch(epoch, vis, train_loader, model, optimizer, criterion, scheduler, opts):
    print('Training of epoch [{}]'.format(epoch))

    model.train()
    tic = time.time()
    train_loader.dataset.transform.transforms[3].set_alpha()
    print("change alpha to ", train_loader.dataset.transform.transforms[3].alpha)

    for i, (images, labels) in enumerate(train_loader):

        images = images.to(int(opts.gpu_ids[opts.rank]))
        labels = labels.to(int(opts.gpu_ids[opts.rank]))
        outputs = model(images)

        # ----------- update -----------
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # get lr
        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        # time
        toc = time.time()

        # visualization
        if (i % opts.vis_step == 0 or i == len(train_loader) - 1) and opts.rank == 0:
            print('Epoch [{0}/{1}], Iter [{2}/{3}], Loss: {4:.4f}, LR: {5:.5f}, Time: {6:.2f}'.format(epoch,
                                                                                                      opts.epoch, i,
                                                                                                      len(train_loader),
                                                                                                      loss.item(),
                                                                                                      lr,
                                                                                                      toc - tic))

            vis.line(X=torch.ones((1, 1)) * i + epoch * len(train_loader),
                     Y=torch.Tensor([loss]).unsqueeze(0),
                     update='append',
                     win='loss',
                     opts=dict(x_label='step',
                               y_label='loss',
                               title='loss',
                               legend=['total_loss']))

    # save pth file
    if opts.rank == 0:

        if not os.path.exists(opts.save_path):
            os.mkdir(opts.save_path)

        checkpoint = {'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'scheduler_state_dict': scheduler.state_dict()}

        torch.save(checkpoint, os.path.join(opts.save_path, opts.save_file_name + '.{}.pth.tar'.format(epoch)))
        print("save .pth")