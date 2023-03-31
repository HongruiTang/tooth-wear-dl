import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torch.nn
from tqdm import tqdm
from model.dataset import JawDataset
from model.model import PointNetReg, feature_transform_regularizer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from statistics import mean 

NUM_POINTS = 2048
BATCH_SIZE = 32

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=BATCH_SIZE, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=NUM_POINTS, help='input batch size')
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--feature_transform', default=True, help="use feature transform")

opt = parser.parse_args(['--dataset', 'drive/MyDrive/Tooth_Wear_Deep_Learning/PLY_dataset'])
print(opt)


blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = JawDataset(root=opt.dataset, npoints=opt.num_points, split='train')
test_dataset = JawDataset(root=opt.dataset, split='test', npoints=opt.num_points)

dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True)

testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True)


num_batch = (len(dataset) / 5 * 4) // opt.batchSize + 1

kfold_loss = []
kfold_acc = []
kfold_v_loss = []
kfold_v_acc = []


for fold, (train_indices, val_indices) in enumerate(kf.split(dataset)):
    train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, train_indices), batch_size=opt.batchSize, shuffle=True)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, val_indices), batch_size=opt.batchSize, shuffle=True)

    classifier = PointNetReg(feature_transform=opt.feature_transform)
    optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    classifier.cuda()

    loss_arr = []
    acc_arr = []
    test_loss = []
    test_acc = []

    for epoch in range(250):
        scheduler.step()
        loss_epoch = torch.zeros(1).cuda()
        acc_epoch = 0
        for i, data in enumerate(train_loader):
            points, target = data
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            classifier = classifier.train()
            pred, trans, trans_feat = classifier(points)

            L = nn.MSELoss()
            loss = L(pred, target)
          
            if opt.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            if i < num_batch - 1:
                loss_epoch += loss
            loss.backward()

            optimizer.step()
            pred_choice = pred.data
            # out = torch.where(pred_choice < 0.5, 0, pred_choice)
            out = torch.where(pred_choice < 1.5, 1, pred_choice)
            out = torch.where((out >= 1.5) & (out < 2.5), 2, out)
            out = torch.where((out >= 2.5) & (out < 3.5), 3, out)
            out = torch.where(out >= 3.5, 4, out)
            correct = out.eq(target.data).cpu().sum()
            if i < num_batch - 1: 
                acc_epoch += correct.item() / float(opt.batchSize)
              # print('accuracy: %f' % (correct.item() / float(opt.batchSize)))
        print('[%d:] train loss: %f accuracy: %f' % (epoch, loss_epoch.item()/(num_batch - 1), acc_epoch / (num_batch - 1)))
        loss_arr.append(loss_epoch.cpu() / (num_batch - 1))
        acc_arr.append(acc_epoch / (num_batch - 1))
        torch.save(classifier.state_dict(), '%s/%d_cls_model_%d.pth' % (opt.outf, fold, epoch))
    kfold_loss.append(loss_arr)
    kfold_acc.append(acc_arr)

    total_correct = 0
    total_testset = 0
    l = torch.zeros(1).cuda()
    with torch.no_grad():
        for i,data in tqdm(enumerate(val_loader, 0)):
            points, target = data
            # target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            L = nn.MSELoss()
            pred, _, _ = classifier(points)
            test_l = L(pred, target)
            l += test_l
            pred_choice = pred.data
            # out = torch.where(pred_choice < 0.5, 0, pred_choice)
            out = torch.where(pred_choice < 1.5, 1, pred_choice)
            out = torch.where((out >= 1.5) & (out < 2.5), 2, out)
            out = torch.where((out >= 2.5) & (out < 3.5), 3, out)
            out = torch.where(out >= 3.5, 4, out)
            correct = out.eq(target.data).cpu().sum()
            total_correct += correct.item()
            total_testset += points.size()[0]
        kfold_v_loss.append(l.cpu() / 38)
        print("final loss {}".format(l.cpu() / 38))
        print(total_correct, total_testset)
        kfold_v_acc.append(total_correct / float(total_testset))
        print("final accuracy {}".format(total_correct / float(total_testset)))



# train loss
fold1_loss = kfold_loss[0]
y = []
for i in fold1_loss:
    i = i.detach().numpy()
    y.append(i)
x = range(1, 251)
plt.plot(x, y)

# Set X and Y labels
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.ylim(0, 2)

plt.show()

# train accuracy
fold1_acc = kfold_acc[0]
x = range(1, 251)
plt.plot(x, fold1_acc)

# Set X and Y labels
plt.title('Traning Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

plt.show()

# validation loss
y = []
for i in kfold_v_loss:
    i = i.detach().numpy()
    y.append(i)
x = ["1st", "2nd", "3rd", "4th", "5th"]
plt.plot(x, y)

# Set X and Y labels
plt.title('Validation Loss')
plt.xlabel('Fold number')
plt.ylabel('Mean Squared Error')
plt.ylim(0, 1)
# plt.xticks(range(1,6,1))

for x, y in zip(x, y):
    label = "{:.4f}".format(y[0])
    plt.text(x, y, label, ha='center', va='bottom')

plt.show()


# validation accuracy
kfold_v_acc = [0.9773218142548596, 0.9697297297297297, 0.9816216216216216, 0.9037837837837838, 0.9156756756756756]
x = ["1st", "2nd", "3rd", "4th", "5th"]
plt.plot(x, kfold_v_acc)

# Set X and Y labels
plt.title('Validation Accuracy')
plt.xlabel('Fold number')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

for x, y in zip(x, kfold_v_acc):
    label = "{:.4f}".format(y)
    plt.text(x, y, label, ha='center', va='top')

plt.show()

####################
tot_mse = []
tot_rmse = []
tot_r2 = []
tot_mae = []

acc = []


for j in range(25):
    k_mse = []
    k_rmse = []
    k_r2 = []
    k_mae = []

    for i in range(5):
        test_model = PointNetReg(feature_transform = True)
        test_model.load_state_dict(torch.load('%s/%d_cls_model_%d.pth' % (opt.outf, i, j*10+9)))
        test_model.to('cuda')

        with torch.no_grad():
            loss_function = nn.MSELoss()
            MAE = nn.L1Loss()
            mse = []
            rmse = []
            r2 =[]
            mae = []
            total_correct = 0
            total_testset = 0
            for i,data in tqdm(enumerate(testdataloader, 0)):
                points, target = data
                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                test_model = test_model.eval()
                pred, _, _ = test_model(points)
              
                loss = loss_function(pred, target)
                # print(target)
                # print(pred)
                # print(loss)
                RMSE_metric = torch.sqrt(loss) # Root Mean Squared Error (RMSE) 
                mae_l1 = MAE(pred, target)
                r_squared = r2_score(target.cpu().numpy(), pred.cpu().numpy())
                mse.append(loss.item())
                mae.append(mae_l1.item())
                rmse.append(RMSE_metric.item())
                r2.append(r_squared)

                # Accuracy
                pred_choice = pred.data
                out = torch.where(pred_choice < 1.5, 1, pred_choice)
                out = torch.where((out >= 1.5) & (out < 2.5), 2, out)
                out = torch.where((out >= 2.5) & (out < 3.5), 3, out)
                out = torch.where(out >= 3.5, 4, out)
                correct = out.eq(target.data).cpu().sum()
                total_correct += correct.item()
                total_testset += points.size()[0]
            print("final accuracy {}".format(total_correct / float(total_testset)))
            acc.append(total_correct / float(total_testset))
            k_mse.append(mse)
            k_rmse.append(rmse)
            k_mae.append(mae)
            k_r2.append(r2)
        print("Test Loss:", sum(mse) / 37)
        print("Test RMSE:", sum(rmse) / 37)
        print("Test R-squared:", sum(r2) / 37)
    tot_mse.append(k_mse)
    tot_rmse.append(k_rmse)
    tot_r2.append(k_r2)
    tot_mae.append(k_mae)

####################
y_values1 = [i[0][0] for i in tot_r2]
y_values2 = [i[1][0] for i in tot_r2]
y_values3 = [i[2][0] for i in tot_r2]
y_values4 = [i[3][0] for i in tot_r2]
y_values5 = [i[4][0] for i in tot_r2]
x_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250]

plt.plot(x_values, y_values1, label='Fold 1')
plt.plot(x_values, y_values2, label='Fold 2')
plt.plot(x_values, y_values3, label='Fold 3')
plt.plot(x_values, y_values4, label='Fold 4')
plt.plot(x_values, y_values5, label='Fold 5')
# Set X and Y labels
plt.title('R^2 of K-fold among epoch 10-250')
plt.xlabel('Epoch')
plt.ylabel('R^2')

plt.ylim(0, 1.4)
plt.xticks(range(0,251,20))
plt.legend()

plt.show()

####################
y_values1 = [i[0][0] for i in tot_r2]
y_values2 = [i[1][0] for i in tot_r2]
y_values3 = [i[2][0] for i in tot_r2]
y_values4 = [i[3][0] for i in tot_r2]
y_values5 = [i[4][0] for i in tot_r2]
x_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250]

plt.plot(x_values, y_values1, label='Fold 1')
plt.plot(x_values, y_values2, label='Fold 2')
plt.plot(x_values, y_values3, label='Fold 3')
plt.plot(x_values, y_values4, label='Fold 4')
plt.plot(x_values, y_values5, label='Fold 5')
# Set X and Y labels
plt.title('R^2 of K-fold among epoch 10-250')
plt.xlabel('Epoch')
plt.ylabel('R^2')

plt.ylim(0, 1.4)
plt.xticks(range(0,251,20))
plt.legend()

plt.show()

####################
f1 = []
f2 = []
f3 = []
f4 = []
f5 = []

for i in range(0, len(acc)):
    if i % 5 == 0:
        f1.append(acc[i])
    elif i % 5 == 1:
        f2.append(acc[i])
    elif i % 5 == 2:
        f3.append(acc[i])
    elif i % 5 == 3:
        f4.append(acc[i])
    else:
        f5.append(acc[i])
  
# Accuracy on test set
x = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250]
plt.plot(x, f5)

# Set X and Y labels
plt.title('Accuracy on test set (fifth model set)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

plt.text(250, f5[-1], "{:.4f}".format(f5[-1]), ha='center', va='bottom')

plt.show()

#####################

y = [mean(f2[10:]), mean(f2[10:]), mean(f3[10:]), mean(f4[10:]), mean(f5[10:])]
x = ["1st", "2nd", "3rd", "4th", "5th"]
plt.plot(x, kfold_v_acc)

# Set X and Y labels
plt.title('Average Accuracy after 100 Epochs')
plt.xlabel('Fold number')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

for x, y in zip(x, y):
    label = "{:.4f}".format(y)
    plt.text(x, y, label, ha='center', va='top')

plt.show()