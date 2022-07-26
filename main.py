import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import numpy as np
import time
import argparse

from model.ResNet import *
from augmentation import feature_augmenation

from dataloader.cifar10 import CIFAR10, CIFAR10_LT
from dataloader.cifar100 import CIFAR100, CIFAR100_LT

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

# os.environ["CUDA_VISIBLE_DEVICES"]= "0"


def parse_args():
    parser = argparse.ArgumentParser(description='latent_space_data_augmentation')

    parser.add_argument('--model', default='ResNet34', type=str) # ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNet32
    parser.add_argument('--datasets', default='cifar10-LT', type=str) # cifar10, cifar10-LT, cifar100, cifar100-LT

    parser.add_argument('--train_batch', default=128, type=int)
    parser.add_argument('--test_batch', default=256, type=int)

    parser.add_argument('--max_epoch', default=200, type=int)
    parser.add_argument('--aug_start_epoch', default=160, type=int)

    parser.add_argument('--learning_rate', default=1e-1, type=int)
    parser.add_argument('--momentum', default=0.9, type=int)
    parser.add_argument('--weight_decay', default=5e-4, type=int)

    parser.add_argument('--k', default=30, type=int)
    parser.add_argument('--n', default=20, type=int)

    parser.add_argument('--run_aug', default=True, type=bool)
    parser.add_argument('--run_train_tsne', default=False, type=bool)
    parser.add_argument('--run_test_tsne', default=False, type=bool)
    parser.add_argument('--run_writer', default=False, type=bool)
    parser.add_argument('--save_weights', default=False, type=bool)

    args = parser.parse_args()

    return args


def get_model_and_datasets(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if args.datasets == 'cifar10':
        dataset = CIFAR10(distributed=False, root='./data', train_batch_size=args.train_batch, test_batch_size=args.test_batch)

        trainloader = dataset.trainloader
        testloader = dataset.testloader

        class_num_list = [5000] * 10
        print("cls_num_list : {}".format(class_num_list))

    elif args.datasets == 'cifar10-LT':
        dataset = CIFAR10_LT(distributed=False, root='./data', train_batch_size=args.train_batch, test_batch_size=args.test_batch)

        trainloader = dataset.trainloader
        testloader = dataset.testloader

        class_num_list = dataset.cls_num_list
        print("cls_num_list : {}".format(class_num_list))

    elif args.datasets == 'cifar100':
        dataset = CIFAR100(distributed=False, root='./data', train_batch_size=args.train_batch, test_batch_size=args.test_batch)

        trainloader = dataset.trainloader
        testloader = dataset.testloader

        class_num_list = [500] * 100
        print("cls_num_list : {}".format(class_num_list))

    elif args.datasets == 'cifar100-LT':
        dataset = CIFAR100_LT(distributed=False, root='./data', train_batch_size=args.train_batch, test_batch_size=args.test_batch)

        trainloader = dataset.trainloader
        testloader = dataset.testloader

        class_num_list = dataset.cls_num_list
        print("cls_num_list : {}".format(class_num_list))

    num_class = len(class_num_list)

    if args.model == 'ResNet18':
        model = ResNet18(num_class=num_class)
    elif args.model == 'ResNet34':
        model = ResNet34(num_class=num_class)
    elif args.model == 'ResNet50':
        model = ResNet50(num_class=num_class)
    elif args.model == 'ResNet101':
        model = ResNet101(num_class=num_class)
    elif args.model == 'ResNet152':
        model = ResNet152(num_class=num_class)
    elif args.model == 'ResNet32':
        model = ResNet32(num_class=num_class)

    model = model.to(device)

    if args.run_aug == True:
        aug_model = feature_augmenation(args, class_num=num_class, last_dim=model.fc.weight.shape[1], class_num_list=class_num_list)
        aug_model = aug_model.to(device)
    else:
        aug_model = None
    
    if args.run_writer == True:
        os.makedirs('./logs', exist_ok=True)
        writer = SummaryWriter('logs/')
    else:
        writer = None
    
    if args.save_weights == True:
        os.makedirs('./weights', exist_ok=True)

    if args.run_train_tsne or args.run_test_tsne:
        os.makedirs('./figure', exist_ok=True)

    return model, aug_model, trainloader, testloader, device, class_num_list, writer


def train_process(epoch, trainloader, model, loss_function, optimizer, scheduler, device, num_class, args, aug_model=None, writer=None):
    model.train()

    train_loss = 0
    train_correct = 0

    train_class_correct = np.zeros(num_class)
    train_class_instance = np.zeros(num_class)

    if aug_model != None:
        new_class_correct = np.zeros(num_class)
        new_class_instance = np.zeros(num_class)

        aug_model.epoch = epoch

    iter_print = int(len(trainloader) / 5)
    start_time = time.time()

    if args.run_train_tsne == True:
        tsne = TSNE(n_components=2, learning_rate='auto', init='random')

    for idx, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs, feature = model(images)

        pred = outputs.argmax(dim=1, keepdims=True)
        pred_correct = pred.eq(labels.view_as(pred))
        train_correct += pred_correct.sum().item()
        for class_idx in range(num_class):
            train_class_correct[class_idx] += pred_correct[labels == class_idx].sum().item()
            train_class_instance[class_idx] += pred_correct[labels == class_idx].shape[0]

        if aug_model != None:
            if epoch >= args.aug_start_epoch:
                aug_model.get_feature_mean_std(feature.detach(), labels)

            if epoch > args.aug_start_epoch:
                new_feature, new_label = aug_model.generate_feature()

                if len(new_feature) != 0:
                    new_output = model.fc(new_feature)
                    outputs = torch.cat((outputs, new_output), 0)
                    feature = torch.cat((feature, new_feature), 0)
                    labels = torch.cat((labels, new_label), 0)

                    new_pred = new_output.argmax(dim=1, keepdims=True)
                    new_correct = new_pred.eq(new_label.view_as(new_pred))
                    for class_idx in range(num_class):
                        new_class_correct[class_idx] += new_correct[new_label == class_idx].sum().item()
                        new_class_instance[class_idx] += new_correct[new_label == class_idx].shape[0]

        if args.run_train_tsne == True and epoch > args.aug_start_epoch:
            if idx % len(trainloader) == 0:
                if aug_model != None:
                    feature = torch.cat((feature, aug_model.feature_mean), 0)

                feature_np = feature.detach().cpu().numpy()
                feature_2d = tsne.fit_transform(feature_np)
                label_np = labels.detach().cpu().numpy()

                batch_size = images.shape[0]

                if aug_model != None:
                    mean_feat = feature_2d[-10:]
                    feature_2d = feature_2d[:-10]

                ori_feats = feature_2d[:batch_size]
                ori_labels = label_np[:batch_size]
                new_feats = feature_2d[batch_size:]
                new_labels = label_np[batch_size:]

                colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
                num_of_class = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
                for label in range(num_class):
                    class_ori_feat = ori_feats[label == ori_labels]
                    class_new_feat = new_feats[label == new_labels]
                    plt.scatter(class_ori_feat[:,0], class_ori_feat[:,1], c=colors[label], marker='o', label=num_of_class[label])
                    plt.scatter(class_new_feat[:,0], class_new_feat[:,1], c=colors[label], marker='*', label=None)

                    if aug_model != None:
                        plt.scatter(mean_feat[label,0], mean_feat[label,1], c=colors[label], marker='^', label=None)

                plt.legend()
                plt.savefig("./figure/{}epoch_{}.jpg".format(epoch, idx))
                plt.clf()

        if epoch > args.aug_start_epoch:
            loss = loss_function(outputs, labels)
        else:
            loss = F.cross_entropy(outputs, labels)

        train_loss += loss

        loss.backward()
        optimizer.step()

        if idx % iter_print == 0:
            print('Train Epoch : {} [{}/{} ({:.0f}%)]\tLoss : {:.6f}\t\ttime : {:.2f}'.format(
                epoch, idx * len(images), len(trainloader.dataset),
                       100 * idx / len(trainloader), loss.item(), time.time() - start_time
            ))

    scheduler.step()

    train_loss /= idx

    if writer != None:
        writer.add_scalar("train_Loss", train_loss, epoch)
        writer.add_scalar("train_Accuracy", 100 * train_correct / len(trainloader.dataset), epoch)

    print('-------------------------------------------------------------------------------------')
    print('Train set - Average Loss : {:.4f}, Accuracy : {}/{} ({:.2f}%)'.format(
        train_loss, train_correct, len(trainloader.dataset), 100 * train_correct / len(trainloader.dataset)))

    train_class_acc = train_class_correct / (train_class_instance + 1e-7)
    print('Train set - Class-wise Accuracy : {}'.format(list(np.round(train_class_acc * 100, 1))))

    if aug_model != None:
        new_class_acc = new_class_correct / (new_class_instance + 1e-7)
        print('Train set - New Feature Accuracy : {}'.format(list(np.round(new_class_acc * 100, 1))))
        print('Train set - New Feature Instance : {}'.format(list(new_class_instance)))


def test_process(epoch, testloader, model, loss_function, device, num_class, args, best_acc_dict=None, aug_model=None, writer=None):
    model.eval()

    test_loss = 0
    test_correct = 0

    test_class_correct = np.zeros(num_class)
    test_class_instance = np.zeros(num_class)

    if args.run_test_tsne == True:
        tsne = TSNE(n_components=2, learning_rate='auto', init='random')

    with torch.no_grad():
        for idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            outputs, feature = model(images)

            test_loss += loss_function(outputs, labels)

            pred = outputs.argmax(dim=1, keepdims=True)
            pred_correct = pred.eq(labels.view_as(pred))
            test_correct += pred_correct.sum().item()
            for class_idx in range(num_class):
                test_class_correct[class_idx] += pred_correct[labels == class_idx].sum().item()
                test_class_instance[class_idx] += pred_correct[labels == class_idx].shape[0]

            if args.run_test_tsne == True and epoch > args.aug_start_epoch:
                if idx == 0:
                    feature_np = feature.detach().cpu().numpy()
                    feature_2d = tsne.fit_transform(feature_np)
                    label_np = labels.detach().cpu().numpy()

                    num_class = 10
                    for label in range(num_class):
                        class_feature = feature_2d[label == label_np]
                        plt.scatter(class_feature[:,0], class_feature[:,1])

                    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
                    plt.savefig("./figure/test_{}epoch.jpg".format(epoch))
                    plt.clf()

    test_loss /= idx
    test_acc = test_correct / len(testloader.dataset)

    if writer != None:
        writer.add_scalar("test_Loss", test_loss, epoch)
        writer.add_scalar("test_Accuracy", 100 * test_acc, epoch)

    print('Test set - Average Loss : {:.4f}, Accuracy : {}/{} ({:.2f}%)'.format(
        test_loss, test_correct, len(testloader.dataset), 100 * test_acc))

    test_class_acc = test_class_correct / test_class_instance
    print('Test set - Class-wise Accuracy : {}'.format(list(np.round(test_class_acc * 100, 1))))

    if best_acc_dict != None and epoch >= args.aug_start_epoch:
        save_best_accuracy_model(args, best_acc_dict, test_acc, epoch, model, aug_model)

    print('-------------------------------------------------------------------------------------')


def save_best_accuracy_model(args, best_acc_dict, test_acc, epoch, model, aug_model):
    if best_acc_dict['best_acc'] < test_acc:
        best_acc_dict['best_acc'] = test_acc
        best_acc_dict['best_epoch'] = epoch

        if args.save_weights == True:
            torch.save(model.state_dict(), './weights/{}_{}.pth'.format(args.datasets, args.model))
            if aug_model != None:
                torch.save(aug_model.state_dict(), './weights/{}_{}_aug.pth'.format(args.datasets, args.model))
            
        print("best model updated!, best_epoch : {}, best_acc : {:.2f}%".format(best_acc_dict['best_epoch'], 100 * best_acc_dict['best_acc']))


def main():
    args = parse_args()

    model, aug_model, trainloader, testloader, device, class_num_list, writer = get_model_and_datasets(args)

    print("batch_size : {}".format(args.train_batch))

    num_class = len(class_num_list)

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[160,180], gamma=0.01)

    beta = 0.9999
    effective_num = 1.0 - np.power(beta, class_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(class_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
    weights = torch.tensor(per_cls_weights).float()

    loss_function = nn.CrossEntropyLoss(weight=weights)

    best_acc_dict = {'best_acc': 0, 'best_epoch': 0}

    for epoch in range(1, args.max_epoch + 1):
        train_process(epoch, trainloader, model, loss_function, optimizer, scheduler, device, num_class, args, aug_model=aug_model, writer=writer)
        test_process(epoch, testloader, model, loss_function, device, num_class, args, best_acc_dict, aug_model=aug_model, writer=writer)

    if args.run_writer == True:
        writer.close()

    print('print best model, epoch : {}, best_acc : {:.2f}%'.format(best_acc_dict['best_epoch'], 100 * best_acc_dict['best_acc']))


if __name__ == '__main__':
    main()