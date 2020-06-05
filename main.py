from __future__ import absolute_import
import time
import torch
import torch.nn as nn
import torch.optim as optim
import csv
from my_module import dataset, alexnet_make, vgg_make, resnet_make, gradcam_make, rand_erase
import os
import torchvision
import glob
from PIL import Image
from torchvision import transforms
# import re
# import argparse
# from torch.optim import lr_scheduler
# import torch.nn.functional as F
# import torchvision
# from torchvision import transforms, datasets, models
# import torch.nn.init as init
# from torchvision.transforms import *
import cv2
import numpy as np
# from matplotlib import pyplot as plt
# import math
import random


cnn_name = 'vgg16_bn'

batchsize = 16
num_classes = 50
num_epochs = 500
original_data_dir = 'manga109_data_50_tvt'
train_csv_name = cnn_name + '_traindata_py.csv'
models_save_name = 'models_' + cnn_name + '_py'
if not os.path.exists(models_save_name):
    os.mkdir(models_save_name)

# net2_dict = './' + models_save_name + '/model_epoch_95.ckpt'
data_dir_test = 'manga109_data_50_tvt/test'
data_dir_train = 'manga109_data_50_tvt/train'
data_dir_val = 'manga109_data_50_tvt/val'
testdata_csv = cnn_name + '_testdata.csv'
testdata_csv_2 = cnn_name + '_testdata_2.csv'

gradcam_save_dir = cnn_name + '_gradcam_save'
if not os.path.exists(gradcam_save_dir):
    os.mkdir(gradcam_save_dir)
gradcam_save_dir_train = gradcam_save_dir + '/train'
if not os.path.exists(gradcam_save_dir_train):
    os.mkdir(gradcam_save_dir_train)
gradcam_save_dir_val = gradcam_save_dir + '/val'
if not os.path.exists(gradcam_save_dir_val):
    os.mkdir(gradcam_save_dir_val)
gradcam_save_dir_test = gradcam_save_dir + '/test'
if not os.path.exists(gradcam_save_dir_test):
    os.mkdir(gradcam_save_dir_test)

gradcam_heatmap_save_dir = cnn_name + '_heatmap_save'
if not os.path.exists(gradcam_heatmap_save_dir):
    os.mkdir(gradcam_heatmap_save_dir)
gradcam_heatmap_save_dir_train = gradcam_heatmap_save_dir + '/train'
if not os.path.exists(gradcam_heatmap_save_dir_train):
    os.mkdir(gradcam_heatmap_save_dir_train)
gradcam_heatmap_save_dir_val = gradcam_heatmap_save_dir + '/val'
if not os.path.exists(gradcam_heatmap_save_dir_val):
    os.mkdir(gradcam_heatmap_save_dir_val)
gradcam_heatmap_save_dir_test = gradcam_heatmap_save_dir + '/test'
if not os.path.exists(gradcam_heatmap_save_dir_test):
    os.mkdir(gradcam_heatmap_save_dir_test)

gradcam_heatmap_gray_save_dir = cnn_name + '_heatmap_gray_save'
if not os.path.exists(gradcam_heatmap_gray_save_dir):
    os.mkdir(gradcam_heatmap_gray_save_dir)
gradcam_heatmap_gray_save_dir_train = gradcam_heatmap_gray_save_dir + '/train'
if not os.path.exists(gradcam_heatmap_gray_save_dir_train):
    os.mkdir(gradcam_heatmap_gray_save_dir_train)
gradcam_heatmap_gray_save_dir_val = gradcam_heatmap_gray_save_dir + '/val'
if not os.path.exists(gradcam_heatmap_gray_save_dir_val):
    os.mkdir(gradcam_heatmap_gray_save_dir_val)
gradcam_heatmap_gray_save_dir_test = gradcam_heatmap_gray_save_dir + '/test'
if not os.path.exists(gradcam_heatmap_gray_save_dir_test):
    os.mkdir(gradcam_heatmap_gray_save_dir_test)

mask_image_save_dir = cnn_name + '_mask_img_save'
if not os.path.exists(mask_image_save_dir):
    os.mkdir(mask_image_save_dir)
mask_image_save_dir_train = mask_image_save_dir + '/train'
if not os.path.exists(mask_image_save_dir_train):
    os.mkdir(mask_image_save_dir_train)
mask_image_save_dir_val = mask_image_save_dir + '/val'
if not os.path.exists(mask_image_save_dir_val):
    os.mkdir(mask_image_save_dir_val)
mask_image_save_dir_test = mask_image_save_dir + '/test'
if not os.path.exists(mask_image_save_dir_test):
    os.mkdir(mask_image_save_dir_test)

num_epochs_re = 30
models_save_re_name = 'models_re_' + cnn_name + '_py'
if not os.path.exists(models_save_re_name):
    os.mkdir(models_save_re_name)
train_re_csv_name = cnn_name + '_traindata_re_py.csv'

testdata_re_csv = cnn_name + '_testdata_re.csv'
testdata_re_csv_2 = cnn_name + '_testdata_re_2.csv'
gradcam_save_dir_re = cnn_name + '_gradcam_save_re'
if not os.path.exists(gradcam_save_dir_re):
    os.mkdir(gradcam_save_dir_re)
gradcam_save_dir_re_train = gradcam_save_dir_re + '/train'
if not os.path.exists(gradcam_save_dir_re_train):
    os.mkdir(gradcam_save_dir_re_train)
gradcam_save_dir_re_val = gradcam_save_dir_re + '/val'
if not os.path.exists(gradcam_save_dir_re_val):
    os.mkdir(gradcam_save_dir_re_val)
gradcam_save_dir_re_test = gradcam_save_dir_re + '/test'
if not os.path.exists(gradcam_save_dir_re_test):
    os.mkdir(gradcam_save_dir_re_test)

cwd = os.getcwd()
# cwd = os.chdir('/home/es1video2_7/workspace/fit_2')
print(cwd)

image_datasets = dataset.data_trans(original_data_dir, cwd)

train_loader, val_loader, test_loader, dataset_sizes, class_names = dataset.loaders(
    batchsize, image_datasets)

inputs, classes = next(iter(train_loader))

out = torchvision.utils.make_grid(inputs)
dataset.imshow(out, title=[class_names[x] for x in classes])

for train in train_loader:
    print("train[0].shape", train[0].shape)
    print("train[0].dtype", train[0].dtype)
    print("type(train[0])", type(train[0]))
    break

device = 'cuda' if torch.cuda.is_available() else 'cpu'


if cnn_name == 'alexnet':
    net = alexnet_make.alexnet(num_classes).to(device)
if cnn_name == 'vgg16_bn':
    net = vgg_make.vgg16_bn(num_classes).to(device)
if cnn_name == 'resnet50':
    net = resnet_make.resnet(num_classes).to(device)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!!")
    net = nn.DataParallel(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01,
                      momentum=0.9, weight_decay=5e-4)
# optimizer = optim.Adam(net.parameters())

# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,60], gamma=0.1)

# lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30], gamma=0.1)

print(net)

# ########################################learning_area####################################
train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []
count = 0
count_lr = 0
avg_val_acc_max = 0
avg_test_acc_max = 0
flag = False


with open(train_csv_name, 'w') as f:
    pass

random_maker = rand_erase.RandomErasing(probability=0.5, mean=[
    0.4914, 0.4822, 0.4465], gd_th=-0.6, sl=0.02, sh=0.4, r1=0.3)


for epoch in range(num_epochs):
    train_loss = 0
    train_acc = 0
    val_loss = 0
    val_acc = 0
    test_loss = 0
    test_acc = 0

    time1 = time.time()

    # train
    net.train()
    # with tqdm(total=len(train_loader)) as pbar:
    for i, (images, labels) in (enumerate(train_loader)):

        ##########################
        # images = random_maker.run_erase_3(images)
        ###########################

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)

        train_loss += loss.detach().item()
    #   print('loss.item()',loss.detach().item())
        train_acc += (outputs.max(1)[1] == labels).sum().item()
        loss.backward()
        optimizer.step()
        print("{}\r".format(i * (batchsize / len(train_loader))), end="")
        # pbar.update(1)
        # pbar.clear()
    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_train_acc = train_acc / len(train_loader.dataset)

    # val
    net.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_acc += (outputs.max(1)[1] == labels).sum().item()
    avg_val_loss = val_loss / len(val_loader.dataset)
    avg_val_acc = val_acc / len(val_loader.dataset)
    if avg_val_acc <= avg_val_acc_max:
        count += 1
    if avg_val_acc > avg_val_acc_max:
        avg_val_acc_max = avg_val_acc
        count = 0
        torch.save(net.module.state_dict(), './' + models_save_name +
                   '/model_epoch_{}.ckpt'.format(epoch + 1))
        no1acc_epoch = str(epoch + 1)

    # test
    net.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            test_acc += (outputs.max(1)[1] == labels).sum().item()
    avg_test_loss = test_loss / len(test_loader.dataset)
    avg_test_acc = test_acc / len(test_loader.dataset)

    if avg_test_acc > avg_test_acc_max:
        avg_test_acc_max = avg_test_acc
        test_max_epoch = str(epoch + 1)

    print('Epoch [{}/{}], loss: {loss:.3f}, val_loss: {val_loss:.3f}, test_loss: {test_loss:.3f}, acc: {acc:.3f}, val_acc: {val_acc:.3f}, test_acc: {test_acc:.3f}, lr:{learning_rate}'
          .format(epoch + 1, num_epochs, i + 1, loss=avg_train_loss, val_loss=avg_val_loss, test_loss=avg_test_loss,
                  acc=avg_train_acc, val_acc=avg_val_acc, test_acc=avg_test_acc, learning_rate=optimizer.param_groups[0]["lr"]))

    ###############################################
    # lr_scheduler.step()
    ###############################################

    train_loss_list.append(avg_train_loss)
    train_acc_list.append(avg_train_acc)
    val_loss_list.append(avg_val_loss)
    val_acc_list.append(avg_val_acc)


    time2 = time.time()
    time3 = time2 - time1
    print("経過時間：", time3)

    with open(train_csv_name, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc,
                         avg_test_acc, optimizer.param_groups[0]["lr"], count, test_max_epoch, time3])

    if count == 3 and count_lr != 3:
        optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * 0.1
        count_lr += 1
        count = 0
        print("lr*0.1")

    if count_lr == 3 and count == 30:
        flag = True
        break

    print("count:", count)

#  ###########################################################gradcam_area###########################
if cnn_name == 'alexnet':
    net2 = alexnet_make.alexnet(num_classes).to(device)
if cnn_name == 'vgg16_bn':
    net2 = vgg_make.vgg16_bn(num_classes).to(device)
if cnn_name == 'resnet50':
    net2 = resnet_make.resnet(num_classes).to(device)

net2_dict = './' + models_save_name + '/model_epoch_' + no1acc_epoch + '.ckpt'
print('net2_dict', net2_dict)

net2.load_state_dict(torch.load(net2_dict), strict=False)

print("net2", net2)

cwd = os.getcwd()
print(cwd)

dir_list_test = sorted(glob.glob(os.path.join(cwd, data_dir_test, '*')))
dir_list_train = sorted(glob.glob(os.path.join(cwd, data_dir_train, '*')))
dir_list_val = sorted(glob.glob(os.path.join(cwd, data_dir_val, '*')))

test_list = []
test_label_list = []
val_list = []
val_label_list = []
train_list = []
train_label_list = []

for i, path in enumerate(dir_list_test):
    image_path_list = sorted(glob.glob(os.path.join(path + '/', '*')))
    test_list += image_path_list
    test_label_list += [i for j in range(len(image_path_list))]

for i, path in enumerate(dir_list_train):
    image_path_list = sorted(glob.glob(os.path.join(path + '/', '*')))
    train_list += image_path_list
    train_label_list += [i for j in range(len(image_path_list))]

for i, path in enumerate(dir_list_val):
    image_path_list = sorted(glob.glob(os.path.join(path + '/', '*')))
    val_list += image_path_list
    val_label_list += [i for j in range(len(image_path_list))]


incorrect_list = []
correct_list = []
all_list = []
count = 1
b = 0
acc = 0
aaa = 0
bbb = 0
Number = len(test_list)
print('Number', Number)

# alexnet
if cnn_name == 'alexnet':
    grad_cam = gradcam_make.GradCam(
        model=net2, target_layer_names=["12"], use_cuda=True)

# vgg16_bn
if cnn_name == 'vgg16_bn':
    grad_cam = gradcam_make.GradCam(
        model=net2, target_layer_names=["43"], use_cuda=True)
        
with open(testdata_csv, 'w') as f:
    pass
with open(testdata_csv_2, 'w') as f:
    pass

tvts = ['train', 'val', 'test']

for tvt in tvts:
    if tvt == 'train':
        img_list = train_list
        label_list = train_label_list

    if tvt == 'val':
        img_list = val_list
        label_list = val_label_list

    if tvt == 'test':
        img_list = test_list
        label_list = test_label_list
        
    for i, (path, label) in enumerate(zip(img_list, label_list)):
        test_image_pil = Image.open(path)

        image_name = os.path.basename(path)

        class_name = os.path.split(path)[-2]
        class_name = os.path.split(class_name)[-1]

        test_image = transforms.Compose([
            transforms.Resize((585, 414), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            # transforms.Normalize([0.5], [0.5])
        ])(test_image_pil)
    #     input_image = torch.zeros((1,3,585,412))
    #     input_image[0,:,:,:] = test_image
        input_image = test_image.unsqueeze(0)
        input_image = input_image.to(device)
        print(input_image.shape)

        target_index = None

    #   for k in range(5):#5 classes
        feature_image = grad_cam(input_image)

        heatmap = cv2.applyColorMap(
            np.uint8(255 * feature_image), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap_gray = np.uint8(255 * feature_image)
        heatmap_gray = np.float32(heatmap_gray) / 255

        # input_image_numpy = np.transpose(input_image.squeeze(0).cpu().numpy(), [1, 2, 0])

        input_image_numpy = cv2.imread(path)
        size = (414, 585)
        input_image_numpy = cv2.resize(input_image_numpy, size)
        cam = heatmap + np.float32(input_image_numpy) / 255
        cam_heatmap = heatmap
        cam_heatmap_gray = heatmap_gray

        cam = cam / np.max(cam)
        cam_heatmap = cam_heatmap / np.max(cam_heatmap)

        cam_heatmap_pil = Image.fromarray(np.uint8(255 * cam_heatmap_gray))
        cam_heatmap_pil = cam_heatmap_pil.convert('L').convert('RGB')
        print(cam_heatmap_gray.shape)
        input_mask_image = transforms.Compose([
            transforms.Resize((585,414), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            # transforms.Normalize([0.5], [0.5])
        ])(cam_heatmap_pil)
        input_mask_image = input_mask_image.unsqueeze(0)
        input_mask_image = input_mask_image.to(device)
        
        random_erasing = rand_erase.RandomErasing(probability = 0.5, mean=[0.4914, 0.4822, 0.4465], gd_th = -0.2, sl = 0.02, sh = 0.4, r1 = 0.3)
        mask_img = random_erasing.mask_generator(input_image, input_mask_image)
        save_all_dir_mask_img = os.path.join(cwd, mask_img_save_dir, tvt,  'class_' + str(label))
        if not os.path.exists(save_all_dir_mask_img):
            os.mkdir(save_all_dir_mask_img)
        torchvision.utils.save_image(mask_img, save_all_dir_mask_img + '/' + image_name + ".jpg", nrow=1, normalize=True)

        save_all_dir = os.path.join(cwd, gradcam_save_dir, tvt,  'class_' + str(label))
        if not os.path.exists(save_all_dir):
            os.mkdir(save_all_dir)

        save_all_dir_heatmap = os.path.join(cwd, gradcam_heatmap_save_dir, tvt,  'class_' + str(label))
        if not os.path.exists(save_all_dir_heatmap):
            os.mkdir(save_all_dir_heatmap)

        save_all_dir_heatmap_gray = os.path.join(cwd, gradcam_heatmap_gray_save_dir, tvt,  'class_' + str(label))
        if not os.path.exists(save_all_dir_heatmap_gray):
            os.mkdir(save_all_dir_heatmap_gray)

        cv2.imwrite(save_all_dir + '/' + image_name + ".jpg", np.uint8(255 * cam))
        cv2.imwrite(save_all_dir_heatmap + '/' + image_name + ".jpg", np.uint8(255 * cam_heatmap))
        cv2.imwrite(save_all_dir_heatmap_gray + '/' + image_name + ".jpg", np.uint8(255 * cam_heatmap_gray))    
  

        if tvt == 'test':
            result = net2(input_image)
            soft_func = nn.Softmax(dim=1)
            result = soft_func(result)
            result_name = str(result)
            result_max = torch.max(result)
            result_max_name = str(result_max)
            result_topk = torch.topk(result, 3)
            topk_name = str(result_topk[0])
            topk_label = str(result_topk[1])

            all_list.append([path, label])

            ans = np.argmax(result.detach().cpu().numpy()[0])

            if(ans == label):
                acc += 1
                correct_list.append([path, label])
                count += 1
                a = ''
                aaa += 1

            else:
                incorrect_list.append([path, label])
                count += 1
                a = 'false'
                b += 1
                bbb += 1

            with open(testdata_csv, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([class_name, image_name, result_max_name[7:13], ans,
                                label, topk_name[7:-40], topk_label[7:-17], a, b, result_name])
            with open(testdata_csv_2, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([class_name, image_name, result_max_name[7:13],
                                ans, label, topk_name[7:-40], topk_label[7:-17], a, b])

print(acc / Number)
print('aaa:', aaa)
print('bbb:', bbb)


##############################################Re_Learning########################################
mask_image_datasets = dataset.data_trans(mask_img_save_dir, cwd)

train_loader_mask, val_loader_mask, test_loader_mask, dataset_sizes, class_names = dataset.loaders(
    batchsize, mask_image_datasets)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!!")
    net2 = nn.DataParallel(net2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.00001,
                      momentum=0.9, weight_decay=5e-4)

val_loss = 0
val_acc = 0

#val
net2.eval()
with torch.no_grad():
  for images, labels in val_loader:
    images = images.to(device)
    labels = labels.to(device)
    outputs = net2(images)
    loss = criterion(outputs, labels)
    val_loss += loss.item()
    val_acc += (outputs.max(1)[1] == labels).sum().item()
avg_val_loss = val_loss / len(val_loader.dataset)
avg_val_acc = val_acc / len(val_loader.dataset)

print ('First test,  val_loss: {val_loss:.6f}, val_acc: {val_acc:.6f}, lr:{learning_rate}'
               .format(val_loss=avg_val_loss, val_acc=avg_val_acc, learning_rate=optimizer.param_groups[0]["lr"]))

train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []
count = 0
count_lr = 0
avg_val_acc_max = 0
avg_test_acc_max = 0
flag = False


with open(train_re_csv_name, 'w') as f:
    pass

for epoch in range(num_epochs_re):
    train_loss = 0
    train_acc = 0
    val_loss = 0
    val_acc = 0
    test_loss = 0
    test_acc = 0

    time1 = time.time()

    # train
    net2.train()
    # with tqdm(total=len(train_loader)) as pbar:
    for i, ((images, labels), (images_mask, _)) in enumerate(zip(train_loader, train_loader_mask)):

        ##########################
        # images = random_maker.run_erase_3(images)
        ###########################

        if random.uniform(0, 1) > 0.5:
            images = images_mask

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net2(images)
        loss = criterion(outputs, labels)

        train_loss += loss.detach().item()
    #   print('loss.item()',loss.detach().item())
        train_acc += (outputs.max(1)[1] == labels).sum().item()
        loss.backward()
        optimizer.step()
        print("{}\r".format(i * (batchsize / len(train_loader))), end="")
        # pbar.update(1)
        # pbar.clear()
    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_train_acc = train_acc / len(train_loader.dataset)

    # val
    net2.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net2(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_acc += (outputs.max(1)[1] == labels).sum().item()
    avg_val_loss = val_loss / len(val_loader.dataset)
    avg_val_acc = val_acc / len(val_loader.dataset)
    if avg_val_acc <= avg_val_acc_max:
        count += 1
    if avg_val_acc > avg_val_acc_max:
        avg_val_acc_max = avg_val_acc
        count = 0
        torch.save(net2.module.state_dict(), './' + models_save_re_name +
            '/model_epoch_{}.ckpt'.format(epoch + 1))
        no1acc_epoch_re = str(epoch + 1)

    # test
    net2.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net2(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            test_acc += (outputs.max(1)[1] == labels).sum().item()
    avg_test_loss = test_loss / len(test_loader.dataset)
    avg_test_acc = test_acc / len(test_loader.dataset)

    if avg_test_acc > avg_test_acc_max:
        avg_test_acc_max = avg_test_acc
        test_max_epoch = str(epoch + 1)

    print('Epoch [{}/{}], loss: {loss:.3f}, val_loss: {val_loss:.3f}, test_loss: {test_loss:.3f}, acc: {acc:.3f}, val_acc: {val_acc:.3f}, test_acc: {test_acc:.3f}, lr:{learning_rate}'
          .format(epoch + 1, num_epochs, i + 1, loss=avg_train_loss, val_loss=avg_val_loss, test_loss=avg_test_loss,
                  acc=avg_train_acc, val_acc=avg_val_acc, test_acc=avg_test_acc, learning_rate=optimizer.param_groups[0]["lr"]))

    ###############################################
    # lr_scheduler.step()
    ###############################################

    train_loss_list.append(avg_train_loss)
    train_acc_list.append(avg_train_acc)
    val_loss_list.append(avg_val_loss)
    val_acc_list.append(avg_val_acc)

    time2 = time.time()
    time3 = time2 - time1
    print("経過時間：", time3)

    with open(train_re_csv_name, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc,
                         avg_test_acc, optimizer.param_groups[0]["lr"], count, test_max_epoch, time3])

    if count == 30:
        flag = True
        break

    print("count:", count)

###################################################gradcam_re#########################
if cnn_name == 'alexnet':
    net3 = alexnet_make.alexnet(num_classes).to(device)
if cnn_name == 'vgg16_bn':
    net3 = vgg_make.vgg16_bn(num_classes).to(device)
if cnn_name == 'resnet50':
    net3 = resnet_make.resnet(num_classes).to(device)

net3_dict = './' + models_save_name + '/model_epoch_' + no1acc_epoch_re + '.ckpt'
print('net3_dict', net3_dict)

net3.load_state_dict(torch.load(net3_dict), strict=False)

print("net3", net3)

# alexnet
if cnn_name == 'alexnet':
    grad_cam = gradcam_make.GradCam(
        model=net3, target_layer_names=["12"], use_cuda=True)

# vgg16_bn
if cnn_name == 'vgg16_bn':
    grad_cam = gradcam_make.GradCam(
        model=net3, target_layer_names=["43"], use_cuda=True)

with open(testdata_re_csv, 'w') as f:
    pass
with open(testdata_re_csv_2, 'w') as f:
    pass

tvts = ['train', 'val', 'test']

for tvt in tvts:
    if tvt == 'train':
        img_list = train_list
        label_list = train_label_list

    if tvt == 'val':
        img_list = val_list
        label_list = val_label_list

    if tvt == 'test':
        img_list = test_list
        label_list = test_label_list
        
    for i, (path, label) in enumerate(zip(img_list, label_list)):
        test_image_pil = Image.open(path)

        image_name = os.path.basename(path)

        class_name = os.path.split(path)[-2]
        class_name = os.path.split(class_name)[-1]

        test_image = transforms.Compose([
            transforms.Resize((585, 414), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            # transforms.Normalize([0.5], [0.5])
        ])(test_image_pil)
    #     input_image = torch.zeros((1,3,585,412))
    #     input_image[0,:,:,:] = test_image
        input_image = test_image.unsqueeze(0)
        input_image = input_image.to(device)
        print(input_image.shape)

        target_index = None

    #   for k in range(5):#5 classes
        feature_image = grad_cam(input_image)

        heatmap = cv2.applyColorMap(
            np.uint8(255 * feature_image), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap_gray = np.uint8(255 * feature_image)
        heatmap_gray = np.float32(heatmap_gray) / 255

        # input_image_numpy = np.transpose(input_image.squeeze(0).cpu().numpy(), [1, 2, 0])

        input_image_numpy = cv2.imread(path)
        size = (414, 585)
        input_image_numpy = cv2.resize(input_image_numpy, size)
        cam = heatmap + np.float32(input_image_numpy) / 255
        cam_heatmap = heatmap
        cam_heatmap_gray = heatmap_gray

        cam = cam / np.max(cam)
        cam_heatmap = cam_heatmap / np.max(cam_heatmap)

        cam_heatmap_pil = Image.fromarray(np.uint8(255 * cam_heatmap_gray))
        cam_heatmap_pil = cam_heatmap_pil.convert('L').convert('RGB')
        print(cam_heatmap_gray.shape)
        input_mask_image = transforms.Compose([
            transforms.Resize((585,414), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            # transforms.Normalize([0.5], [0.5])
        ])(cam_heatmap_pil)
        input_mask_image = input_mask_image.unsqueeze(0)
        input_mask_image = input_mask_image.to(device)
        
        save_all_dir = os.path.join(cwd, gradcam_save_dir_re, tvt,  'class_' + str(label))
        if not os.path.exists(save_all_dir):
            os.mkdir(save_all_dir)

        cv2.imwrite(save_all_dir + '/' + image_name + ".jpg", np.uint8(255 * cam))  

        if tvt == 'test':
            result = net3(input_image)
            soft_func = nn.Softmax(dim=1)
            result = soft_func(result)
            result_name = str(result)
            result_max = torch.max(result)
            result_max_name = str(result_max)
            result_topk = torch.topk(result, 3)
            topk_name = str(result_topk[0])
            topk_label = str(result_topk[1])

            all_list.append([path, label])

            ans = np.argmax(result.detach().cpu().numpy()[0])

            if(ans == label):
                acc += 1
                correct_list.append([path, label])
                count += 1
                a = ''
                aaa += 1

            else:
                incorrect_list.append([path, label])
                count += 1
                a = 'false'
                b += 1
                bbb += 1

            with open(testdata_re_csv, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([class_name, image_name, result_max_name[7:13], ans,
                                label, topk_name[7:-40], topk_label[7:-17], a, b, result_name])
            with open(testdata_re_csv_2, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([class_name, image_name, result_max_name[7:13],
                                ans, label, topk_name[7:-40], topk_label[7:-17], a, b])

print(acc / Number)
print('aaa:', aaa)
print('bbb:', bbb)