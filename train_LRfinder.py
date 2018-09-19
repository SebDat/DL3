import socket
import timeit
from datetime import datetime
import os
import glob
from collections import OrderedDict

# PyTorch includes
import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid


# Tensorboard include
from tensorboardX import SummaryWriter

# Custom includes
from dataloaders import pascal, sbd, combine_dbs
from dataloaders import utils
from networks import deeplab_xception, deeplab_resnet
from dataloaders import custom_transforms as tr

from tqdm import tqdm
from datetime import datetime

gpu_id = 0
print('Using GPU: {} '.format(gpu_id))
# Setting parameters
use_sbd = False  #  True  # Whether to use SBD dataset
nEpochs = 1 # 100  # Number of epochs for training
resume_epoch = 0   # Default is 0, change if want to resume

p = OrderedDict()  # Parameters to include in report
p['trainBatch'] = 2  # Training batch size
testBatch = 2  # Testing batch size
useTest = True  # See evolution of the test set when training
nTestInterval = 5 # Run on test set every nTestInterval epochs
snapshot = 5  # Store a model every snapshot epochs
p['nAveGrad'] = 1  # Average the gradient of several iterations
p['lr'] = 1e-8  # Learning rate
p['wd'] = 5e-4  # Weight decay
p['momentum'] = 0.9  # Momentum
p['epoch_size'] = 10  # How many epochs to change learning rate
backbone = 'resnet'  #'xception' # Use xception or resnet as feature extractor,

#Learning Rate Restart Parameters
cycle_len = 2    #number of epoch between restart
#cycle_mult = 1   #
annealing = 1

number_ex = 10  #how many examples to see before increasing learning rate

def get_lr(lr_base, current_epoch, current_batch, tot_batch, cycle_len, annealing = 1):
    factor = 0.1
    #period_len = tot_batch*cycle_len*cycle_mult**(current_epoch//cycle_len)
    lr_base = lr_base*annealing**(current_epoch//cycle_len)
    period_len = tot_batch*cycle_len
    current_batch_period = current_batch + tot_batch*(current_epoch-cycle_len*(current_epoch//cycle_len))
    lr = 0.5*(1-factor)*lr_base*(np.cos(np.pi*(current_batch_period/period_len))+1)+factor*lr_base
    return lr

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))

# Network definition
if backbone == 'xception':
    net = deeplab_xception.DeepLabv3_plus(nInputChannels=3, n_classes=21, os=16, pretrained=True)
elif backbone == 'resnet':
    net = deeplab_resnet.DeepLabv3_plus(nInputChannels=3, n_classes=21, os=16, pretrained=True)
else:
    raise NotImplementedError

modelName = 'deeplabv3plus-' + backbone + '-voc'
criterion = utils.cross_entropy2d


if resume_epoch == 0:
    print("Training deeplabv3+ from scratch...")
else:
    print("Initializing weights from: {}...".format(
        os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth')))
    net.load_state_dict(
        torch.load(os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth'),
                   map_location=lambda storage, loc: storage)) # Load all tensors onto the CPU

if gpu_id >= 0:
    torch.cuda.set_device(device=gpu_id)
    net.cuda()

if resume_epoch != nEpochs:
    # Logging into Tensorboard
    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    # Use the following optimizer
    optimizer = optim.SGD(net.parameters(), lr=p['lr'], momentum=p['momentum'], weight_decay=p['wd'])
    p['optimizer'] = str(optimizer)

    composed_transforms_tr = transforms.Compose([
        tr.RandomSized(512),
        tr.RandomRotate(15),
        tr.RandomHorizontalFlip(),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])

    composed_transforms_ts = transforms.Compose([
        tr.FixedResize(size=(512, 512)),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])

    voc_train = pascal.VOCSegmentation(split='train', transform=composed_transforms_tr)
    voc_val = pascal.VOCSegmentation(split='val', transform=composed_transforms_ts)

    if use_sbd:
        print("Using SBD dataset")
        sbd_train = sbd.SBDSegmentation(split=['train', 'val'], transform=composed_transforms_tr)
        db_train = combine_dbs.CombineDBs([voc_train, sbd_train], excluded=[voc_val])
    else:
        db_train = voc_train

    trainloader = DataLoader(db_train, batch_size=p['trainBatch'], shuffle=True, num_workers=0)  #num_workers=0
    testloader = DataLoader(voc_val, batch_size=testBatch, shuffle=False, num_workers=0)

    utils.generate_param_report(os.path.join(save_dir, exp_name + '.txt'), p)

    num_img_tr = len(trainloader)
    num_img_ts = len(testloader)
    running_loss_tr = 0.0
    running_loss_ts = 0.0
    aveGrad = 0
    global_step = 0
    print("Training Network")

    tot_batch = int(len(voc_train.images)/p['trainBatch'])
    counter = - p['trainBatch']
    # Main Training and Testing Loop
    for epoch in tqdm(range(resume_epoch, nEpochs)):
        start_time = timeit.default_timer()
        print('\n\n')
        print("Optimization step number {0} test started at {1}".format(epoch+1,datetime.now()))
        print('\n\n')
        
        #if epoch % p['epoch_size'] == p['epoch_size'] - 1:
        #    lr_ = utils.lr_poly(p['lr'], epoch, nEpochs, 0.9)
        #    print('(poly lr policy) learning rate: ', lr_)
        #    optimizer = optim.SGD(net.parameters(), lr=lr_, momentum=p['momentum'], weight_decay=p['wd'])
        LOSS = []
        net.train()
        for ii, sample_batched in enumerate(trainloader):

            counter += p['trainBatch']

            lr_ = p['lr']*2**(counter//number_ex)

            optimizer = optim.SGD(net.parameters(), lr=lr_, momentum=p['momentum'], weight_decay=p['wd'])

            inputs, labels = sample_batched['image'], sample_batched['label']
            # Forward-Backward of the mini-batch
            inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
            global_step += inputs.data.shape[0]

            if gpu_id >= 0:
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = net.forward(inputs)

            loss = criterion(outputs, labels, size_average=False, batch_average=True)
            running_loss_tr += loss.item()

            # Print and store loss
            if ii % number_ex == (number_ex - 1):
                running_loss_tr = running_loss_tr / number_ex
                writer.add_scalar('data/total_loss_epoch', running_loss_tr, epoch)
                print('[Epoch: %d, numImages: %5d]' % (epoch, ii * p['trainBatch'] + inputs.data.shape[0]))
                print('Loss: %f' % running_loss_tr)
                LOSS.append(running_loss_tr)
                running_loss_tr = 0
                stop_time = timeit.default_timer()
                print("Execution time: " + str(stop_time - start_time) + "\n")
                
            # Backward the averaged gradient
            loss /= p['nAveGrad']
            loss.backward()
            aveGrad += 1

            # Update the weights once in p['nAveGrad'] forward passes
            if aveGrad % p['nAveGrad'] == 0:
                writer.add_scalar('data/total_loss_iter', loss.item(), ii + num_img_tr * epoch)
                optimizer.step()
                optimizer.zero_grad()
                aveGrad = 0

    plt.plot(LOSS,'-bo')
    plt.show()
    writer.close()
