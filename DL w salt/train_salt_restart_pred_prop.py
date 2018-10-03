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
from dataloaders import pascal, sbd, combine_dbs, salt_id, salt_id2
from dataloaders import utils, FocalLoss 
from networks import deeplab_xception2, deeplab_resnet
from dataloaders import custom_transforms as tr

from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


gpu_id = 0
print('Using GPU: {} '.format(gpu_id))
# Setting parameters
use_sbd = False  # Whether to use SBD dataset
nEpochs = 1000  # Number of epochs for training
resume_epoch =640   # Default is 0, change if want to resume

p = OrderedDict()  # Parameters to include in report
p['trainBatch'] = 16  # Training batch size
testBatch = 16  # Testing batch size
useTest = True  # See evolution of the test set when training
nTestInterval = 5 # Run on test set every nTestInterval epochs
snapshot = 10  # Store a model every snapshot epochs
p['nAveGrad'] = 1  # Average the gradient of several iterations
p['lr'] = 3e-5  # Learning rate 1e-5
p['wd'] = 1e-3  # Weight decay ini: 5e-4
p['momentum'] = 0.9  # Momentum
p['epoch_size'] = 5  # How many epochs to change learning rate
backbone = 'xception' # Use xception or resnet as feature extractor,
sz = 128  #image size
LossFunc = 'focal'

TrainValSplit = False   #if True, split the training dataset into train/val


#Learning Rate Restart Parameters
cycle_len = 2    #number of epoch between restart
#cycle_mult = 1   #
annealing = 0.98

number_ex = 40  #how many examples to see before increasing learning rate

def get_lr(lr_base, current_epoch, current_batch, tot_batch, cycle_len, resume_epoch, annealing = 1):
    factor = 0.1
    current_epoch = current_epoch - resume_epoch
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
    net = deeplab_xception2.DeepLabv3_plus(nInputChannels=3, n_classes=2, os=16, pretrained=True)
elif backbone == 'resnet':
    net = deeplab_resnet.DeepLabv3_plus(nInputChannels=3, n_classes=2, os=16, pretrained=True)
else:
    raise NotImplementedError

modelName = 'deeplabv3plus-' + backbone + '-voc'
criterion = utils.cross_entropy2d

criterionfocal = FocalLoss.FocalLoss(gamma=2, alpha=None, size_average=False)


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

# Compute the statistics of the dataset of images
location = 'home'

if(location == 'home'):
    PATH = '/home/katou/Python/GitHubRepo/Data/Kaggle Salt Id/'
elif(location == 'work'):
    PATH = 'C:\\Users\\SCatheline\\Documents\\GitHub repo\\FirstTest\\Kaggle_Challenge_LIVE-master\\data\\'
else:
    print('Unavailable location.')

#train image + mask data
train_mask = pd.read_csv(PATH+'train.csv')
#depth data
Depths = pd.read_csv(PATH+'depths.csv')
#salt data
SaltProp = pd.read_csv(PATH+ 'salt_prop.csv')
#training path
train_path = PATH+'train'
#list of files
file_list = list(train_mask['id'].values)

train_ids = next(os.walk(train_path+"/images"))[2] if location == 'home' else next(os.walk(train_path+"\\images"))[2]

im_chan = 1
im_width = sz
im_height = sz
n_features = 1 # Number of extra features, like depth
border = 2

if(TrainValSplit):

    # Get and resize train images and masks
    X = np.zeros((len(train_ids), im_height, im_width, im_chan), dtype=np.float32)
    M = np.zeros((len(train_ids),), dtype=np.float32)
    y = np.zeros((len(train_ids), im_height, im_width, 1), dtype=np.float32)
    X_feat = np.zeros((len(train_ids), n_features), dtype=np.float32)
    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = train_path 
        # Depth
        #X_feat[n] = depth.loc[id_.replace('.png', ''), 'z']
        
        # Load X
        img = load_img(path + '/images/' + id_, grayscale=True) if location == 'home' else load_img(path + '\\images\\' + id_, grayscale=True)
        x_img = img_to_array(img)
        x_img = resize(x_img, (sz, sz, 1), mode='constant', preserve_range=True)
        
        # Create cumsum x
        x_center_mean = x_img[border:-border, border:-border].mean()
        x_csum = (np.float32(x_img)-x_center_mean).cumsum(axis=0)
        x_csum -= x_csum[border:-border, border:-border].mean()
        x_csum /= max(1e-3, x_csum[border:-border, border:-border].std())

        # Load Y
        mask = img_to_array(load_img(path + '/masks/' + id_, grayscale=True)) if location == 'home' else img_to_array(load_img(path + '\\masks\\' + id_, grayscale=True))
        mask = resize(mask, (sz, sz, 1), mode='constant', preserve_range=True)

        # Save images
        X[n, ..., 0] = x_img.squeeze() / 255
        M[n] = np.mean(x_img.squeeze() / 255)
        #X[n, ..., 1] = x_csum.squeeze()
        y[n] = mask / 255

    print('Done!')

    #split the data using the coverage of salt in the image
    Cov = np.zeros((y.shape[0],),dtype = float)
    for i in range(y.shape[0]):
        Cov[i] = int(round(10*np.mean(y[i,:,:,0])))
    del X
    del y
    del X_feat

    MM = np.mean(M)
    SS = np.std(M)

    x_names = np.array([o.split('.')[0] for o in train_ids])
    y_names = np.array([o.split('.')[0] for o in train_ids])
    trn_x, val_x, trn_y, val_y = train_test_split(x_names, y_names, test_size=0.15, stratify=Cov, random_state=42)

    f = open(train_path+'/train.txt', "w+")
    for i in range(len(trn_x)):
        f.write(trn_x[i]+'\n')
    f.close()

    f = open(train_path+'/val.txt', "w+")
    for i in range(len(val_x)):
        f.write(val_x[i]+'\n')
    f.close()
else:
    df_train = pd.read_csv(train_path+'/train.txt')
    trn_x = df_train.values.squeeze()
    df_val = pd.read_csv(train_path+'/val.txt')
    val_x = df_val.values.squeeze()


#Normalize depths data
depth_min = 1e6
depth_max = 0
for i in range(len(trn_x)):
    depth_min = min(Depths[Depths['id'] == trn_x[i]].values[0,1],depth_min)
    depth_max = max(Depths[Depths['id'] == trn_x[i]].values[0,1],depth_max)



# Training loop
if resume_epoch != nEpochs:
    # Logging into Tensorboard
    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    # Use the following optimizer
    optimizer = optim.SGD(net.parameters(), lr=p['lr'], momentum=p['momentum'], weight_decay=p['wd'])
    p['optimizer'] = str(optimizer)

    composed_transforms_tr = transforms.Compose([
        tr.RandomSized(sz),
        tr.RandomRotate(15),
        tr.RandomHorizontalFlip(),
        tr.RandomVerticalFlip(),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])

    composed_transforms_ts = transforms.Compose([
        tr.FixedResize(size=(sz, sz)),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])

    voc_train = salt_id2.SALT2Segmentation(split='train', transform=composed_transforms_tr)
    voc_val = salt_id2.SALT2Segmentation(split='val', transform=composed_transforms_ts)

    if use_sbd:
        print("Using SBD dataset")
        sbd_train = sbd.SBDSegmentation(split=['train', 'val'], transform=composed_transforms_tr)
        db_train = combine_dbs.CombineDBs([voc_train, sbd_train], excluded=[voc_val])
    else:
        db_train = voc_train

    trainloader = DataLoader(db_train, batch_size=p['trainBatch'], shuffle=True, num_workers=0)
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
    lr_ = p['lr']
    # Main Training and Testing Loop
    for epoch in tqdm(range(resume_epoch, nEpochs)):
        start_time = timeit.default_timer()
        print('Epoch number {0} started at {1}'.format(epoch+1, datetime.now()))

        #if epoch % p['epoch_size'] == p['epoch_size'] - 1:
        #    lr_ = utils.lr_poly(p['lr'], epoch, nEpochs, 0.9)
        #    print('(poly lr policy) learning rate: ', lr_)
        #    optimizer = optim.SGD(net.parameters(), lr=lr_, momentum=p['momentum'], weight_decay=p['wd'])
        writer.add_scalar('data/learning_rate_epoch', lr_, epoch)
        
        net.train()
        for ii, sample_batched in enumerate(trainloader):

            lr_ = get_lr(p['lr'], epoch, ii, tot_batch, cycle_len, resume_epoch, annealing)
            optimizer = optim.SGD(net.parameters(), lr=lr_, momentum=p['momentum'], weight_decay=p['wd'])
            
            optimizer = optim.Adam(net.parameters(), lr=lr_)

            inputs, labels, salt_props, depths = sample_batched['image'], sample_batched['label'], sample_batched['salt proportion'], sample_batched['depth']
            #print('Inputs size: {0}'.format(inputs.size()))
            #print('Labels size: {0}'.format(labels.size()))

            b = labels.numpy()
            b = np.round(b/(2**16-1))
            b = b.astype(int)
            c=b.reshape(-1,1)

            labels = torch.from_numpy(b)

            depths = (depths-depth_min)/(depth_max-depth_min)


            depths = torch.tensor(depths,dtype=torch.float64) 

            # Forward-Backward of the mini-batch
            inputs, labels, salt_props, depths  = Variable(inputs, requires_grad=True), Variable(labels), Variable(salt_props, requires_grad=True), Variable(depths, requires_grad=True)
            global_step += inputs.data.shape[0]

            if gpu_id >= 0:
                inputs, labels, salt_props, depths = inputs.cuda(), labels.cuda(), salt_props.cuda(), depths.cuda()

            outputs = net.forward([inputs, salt_props, depths])

            if(LossFunc == 'crossentropy'):
                loss = criterion(outputs, labels, size_average=False, batch_average=True)
            elif(LossFunc == 'focal'):
                loss = criterionfocal.forward(outputs, labels)
            else:
                print('Unimplemented loss.')

            running_loss_tr += loss.item()

            # Print stuff
            if ii % num_img_tr == (num_img_tr - 1):
                running_loss_tr = running_loss_tr / num_img_tr
                writer.add_scalar('data/total_loss_epoch', running_loss_tr, epoch)
                print('[Epoch: %d, numImages: %5d]' % (epoch, ii * p['trainBatch'] + inputs.data.shape[0]))
                print('Loss: %f' % running_loss_tr)
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

            # Show 10 * 3 images results each epoch
            if ii % (num_img_tr // 10) == 0:
                grid_image = make_grid(inputs[:3].clone().cpu().data, 3, normalize=True)
                writer.add_image('Image', grid_image, global_step)
                grid_image = make_grid(utils.decode_seg_map_sequence(torch.max(outputs[:3], 1)[1].detach().cpu().numpy()), 3, normalize=False,
                                       range=(0, 255))
                writer.add_image('Predicted label', grid_image, global_step)
                grid_image = make_grid(utils.decode_seg_map_sequence(torch.squeeze(labels[:3], 1).detach().cpu().numpy()), 3, normalize=False, range=(0, 255))
                writer.add_image('Groundtruth label', grid_image, global_step)

        # Save the model
        if (epoch % snapshot) == snapshot - 1:
            torch.save(net.state_dict(), os.path.join(save_dir, 'models', modelName + '_epoch-' + str(epoch) + '.pth'))
            print("Save model at {}\n".format(os.path.join(save_dir, 'models', modelName + '_epoch-' + str(epoch) + '.pth')))

        # One testing epoch
        if useTest and epoch % nTestInterval == (nTestInterval - 1):
            total_miou = 0.0
            net.eval()
            for ii, sample_batched in enumerate(testloader):
                inputs, labels, salt_props, depths = sample_batched['image'], sample_batched['label'], sample_batched['salt proportion'], sample_batched['depth']

                b = labels.numpy()
                b = np.round(b/(2**16-1))
                b = b.astype(int)
                c=b.reshape(-1,1)

                labels = torch.from_numpy(b)
                depths = (depths-depth_min)/(depth_max-depth_min)
                depths = torch.tensor(depths,dtype=torch.float64)
                
                # Forward-Backward of the mini-batch
                inputs, labels, salt_props, depths  = Variable(inputs, requires_grad=True), Variable(labels), Variable(salt_props, requires_grad=True), Variable(depths, requires_grad=True)
                global_step += inputs.data.shape[0]

                if gpu_id >= 0:
                    inputs, labels, salt_props, depths = inputs.cuda(), labels.cuda(), salt_props.cuda(), depths.cuda()

                with torch.no_grad():
                    outputs = net.forward([inputs, salt_props, depths])

                predictions = torch.max(outputs, 1)[1]

                if(LossFunc == 'crossentropy'):
                    loss = criterion(outputs, labels, size_average=False, batch_average=True)
                elif(LossFunc == 'focal'):
                    loss = criterionfocal.forward(outputs, labels)
                else:
                    print('Unimplemented loss.')

                running_loss_ts += loss.item()

                total_miou += utils.get_iou(predictions, labels)

                # Print stuff
                if ii % num_img_ts == num_img_ts - 1:

                    miou = total_miou / (ii * testBatch + inputs.data.shape[0])
                    running_loss_ts = running_loss_ts / num_img_ts

                    print('Validation:')
                    print('[Epoch: %d, numImages: %5d]' % (epoch, ii * testBatch + inputs.data.shape[0]))
                    writer.add_scalar('data/test_loss_epoch', running_loss_ts, epoch)
                    writer.add_scalar('data/test_miour', miou, epoch)
                    print('Loss: %f' % running_loss_ts)
                    print('MIoU: %f\n' % miou)
                    running_loss_ts = 0


    writer.close()


print('Starting evaluating the model..')

# Evaluation on testing dataset
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

composed_transforms_tr = transforms.Compose([
        tr.RandomSized(128),
        tr.RandomRotate(15),
        tr.RandomHorizontalFlip(),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])

composed_transforms_ts = transforms.Compose([
        tr.FixedResize(size=(128, 128)),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])

voc_train = salt_id2.SALT2Segmentation(split='train', transform=composed_transforms_tr)
voc_val = salt_id2.SALT2Segmentation(split='val', transform=composed_transforms_ts)

testloader = DataLoader(voc_val, batch_size=testBatch, shuffle=False, num_workers=0)   

# Start Test
total_miou = 0.0
net.eval()

num_img_ts = len(testloader)*testBatch

LabelsNumpy = np.zeros((num_img_ts,sz,sz))
PredNumpy = np.zeros((num_img_ts,sz,sz))
cc=0
for ii, sample_batched in enumerate(testloader):
    
    inputs, labels, salt_props, depths = sample_batched['image'], sample_batched['label'], sample_batched['salt proportion'], sample_batched['depth']

    b = labels.numpy()
    b = np.round(b/(2**16-1))
    b = b.astype(int)
    c=b.reshape(-1,1)

    labels = torch.from_numpy(b)
    depths = (depths-depth_min)/(depth_max-depth_min)
    depths = torch.tensor(depths,dtype=torch.float64)

    # Forward-Backward of the mini-batch
    inputs, labels, salt_props, depths  = Variable(inputs, requires_grad=True), Variable(labels), Variable(salt_props, requires_grad=True), Variable(depths, requires_grad=True)
    global_step += inputs.data.shape[0]

    if gpu_id >= 0:
        inputs, labels, salt_props, depths = inputs.cuda(), labels.cuda(), salt_props.cuda(), depths.cuda()

    with torch.no_grad():
        outputs = net.forward([inputs, salt_props, depths])

    predictions = torch.max(outputs, 1)[1]

    if(LossFunc == 'crossentropy'):
        loss = criterion(outputs, labels, size_average=False, batch_average=True)
    elif(LossFunc == 'focal'):
        loss = criterionfocal.forward(outputs, labels)
    else:
        print('Unimplemented loss.')

    #store the labels + outputs into numpy array for later use
    LabelsNumpy[cc:cc+labels.size(0),:,:] = labels.cpu().numpy()[:,0,:,:]
    PredNumpy[cc:cc+labels.size(0),:,:] = outputs.cpu().numpy()[:,1,:,:]

    running_loss_ts += loss.item()

    total_miou += utils.get_iou(predictions, labels)

    # Print stuff
    if ii % num_img_ts == num_img_ts - 1:

        miou = total_miou / (ii * testBatch + inputs.data.shape[0])
        running_loss_ts = running_loss_ts / num_img_ts

        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, ii * testBatch + inputs.data.shape[0]))
        writer.add_scalar('data/test_loss_epoch', running_loss_ts, epoch)
        writer.add_scalar('data/test_miour', miou, epoch)
        print('Loss: %f' % running_loss_ts)
        print('MIoU: %f\n' % miou)
        running_loss_ts = 0                
    cc+=testBatch


#use competition IoU metric
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in
    
    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)   


y_valid = LabelsNumpy
pred = PredNumpy 

IOU = iou_metric_batch(np.round(y_valid).astype(int).squeeze(),np.round(pred).astype(int).squeeze())
print('IOU metric: {0}'.format(np.mean(IOU)))  

thresholds = np.linspace(0, 1, 50)
ious = np.array([iou_metric_batch(y_valid, np.int32(pred > threshold)) for threshold in thresholds])   

plt.plot(thresholds,ious,'-xb')

threshold_best_index = np.argmax(ious[9:-10]) + 9
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]
print('Best IOU score: {0}, threshold used: {1}'.format(iou_best,threshold_best))
