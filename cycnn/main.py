import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.autograd as autograd

import torchvision
import torchvision.transforms as transforms

import cv2 as cv
import numpy as np

import sys, os, argparse, time, random

from models.getmodel import get_model

from PIL import Image

from data import load_data

import utils, image_transforms

import pickle

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm, classes, save_path):
    """ Plot and save the confusion matrix as an image """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()


def save_confusion_matrix(cm, train_set, test_set, output_dir):
    cm_path = os.path.join(output_dir, f"confusion_matrix_{train_set}_test_on_{test_set}.npy")
    np.save(cm_path, cm)
    plot_confusion_matrix(cm, classes=list(range(10)), save_path=cm_path.replace('.npy', '.png'))
    print(f"Confusion Matrix saved as: {cm_path} and {cm_path.replace('.npy', '.png')}")


def train(model, device, optimizer, criterion, train_loader, epoch, args):
    model.train()
    train_loss = 0

    for batch_idx, (images, labels) in enumerate(train_loader):

        """Resize images to fit into the model"""
        images = image_transforms.resize_images(images, 32, 32)

        """Apply image transforms"""
        if args['augmentation'] is not None and 'scale' in args['augmentation']:
            images = image_transforms.random_scale(images)
        if args['augmentation'] is not None and 'rot' in args['augmentation']:
            images = image_transforms.random_rotate(images)
        if args['augmentation'] is not None and 'trans' in args['augmentation']:
            images = image_transforms.random_translate(images)

        """Apply polar mapping"""
        if args['polar_transform'] is not None:
            images = image_transforms.polar_transform(images, transform_type=args['polar_transform'])

        images, labels = images.to(device), labels.to(device)
        result = model(images)

        if args['model'] == 'hnet':
            result = result.sum(dim=(2, 3))

        loss = criterion(result, labels)

        train_loss += loss.item()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    """Print training summary"""
    print('[Epoch {}] Train Loss: {:.6f}'.format(
        epoch, train_loss / len(train_loader)))

    return train_loss


def validate(model, device, criterion, test_loader, epoch, args):
    model.eval()
    validation_loss, correct, num_data = 0, 0, 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):

            if args['dataset'] == 'svhn':
                labels[labels == 9] = 6

            """Resize images to fit into the model"""
            images = image_transforms.resize_images(images, 32, 32)

            """Apply image transforms"""
            if args['augmentation'] is not None and 'scale' in args['augmentation']:
                images = image_transforms.random_scale(images)
            if args['augmentation'] is not None and 'rot' in args['augmentation']:
                images = image_transforms.random_rotate(images)
            if args['augmentation'] is not None and 'trans' in args['augmentation']:
                images = image_transforms.random_translate(images)

            """Apply polar transforms"""
            if args['polar_transform'] is not None:
                images = image_transforms.polar_transform(images, transform_type=args['polar_transform'])

            images, labels = images.to(device), labels.to(device)
            result = model(images)

            if args['model'] == 'hnet':
                result = result.sum(dim=(2, 3))

            loss = criterion(result, labels)

            validation_loss += loss.float()
            pred = result.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            pred = pred.view(pred.size()[0])

            correct += pred.eq(labels.view_as(pred)).sum().item()
            num_data += len(images)

    validation_loss /= len(test_loader)
    accuracy = 100. * correct / num_data

    """Print Validation Summary"""
    print('[Epoch {}] Validation loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        epoch, validation_loss,
        correct, num_data, accuracy))

    return validation_loss, accuracy


# def test(model, device, criterion, test_loader, args, output_dir, train_set, test_set):
def test(model, device, criterion, test_loader, args, output_dir):
    model.eval()
    test_loss, correct, num_data = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):

            if args['dataset'] == 'svhn':
                labels[labels == 9] = 6

            """Resize images to fit into the model"""
            images = image_transforms.resize_images(images, 32, 32)

            """Random rotation is always applied at testing"""
            images = image_transforms.random_rotate(images)

            """Apply polar transforms"""
            if args['polar_transform'] is not None:
                images = image_transforms.polar_transform(images, transform_type=args['polar_transform'])

            images, labels = images.to(device), labels.to(device)
            result = model(images)

            loss = criterion(result, labels)
            test_loss += loss.float()  # sum up batch loss
            pred = result.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            pred = pred.view(pred.size()[0])

            correct += pred.eq(labels.view_as(pred)).sum().item()
            num_data += len(images)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss /= len(test_loader)
    accuracy = 100. * correct / num_data

    """Print Test Summary"""
    print('Test loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, num_data, accuracy))
    # num_classes = len(np.unique(all_labels))
    # cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    # # cm = confusion_matrix(all_labels, all_preds, labels=list(range(10)))
    # cm_path = f"confusion_matrix_{args['dataset']}_{args['model']}.npy"
    # np.save(cm_path, cm)

    # plot_confusion_matrix(cm, classes=list(range(10)), save_path=cm_path.replace('.npy', '.png'))
    # print(f"Confusion Matrix saved as: {cm_path} and {cm_path.replace('.npy', '.png')}")

    # save_confusion_matrix(all_labels, all_preds, train_set, test_set, args['output_dir'])
    # cm = confusion_matrix(all_labels, all_preds)
    # save_confusion_matrix(cm, train_set, test_set, output_dir)
    
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(10)))

    cm_file = os.path.join(output_dir, 'confusion_matrix.npy')
    np.save(cm_file, cm)
    plot_confusion_matrix(cm, classes=list(range(10)), save_path=cm_file.replace('.npy', '.png'))
    print(f"Confusion Matrix saved as: {cm_file} and {cm_file.replace('.npy', '.png')}")

    return test_loss, accuracy


def main():

    ###################################################################
    ## argument
    ###################################################################

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--model', type=str, default='resnet20', help='Model to train.')
    parser.add_argument('--model-path', type=str, default=None, help='Path to the trained model file.')
    parser.add_argument('--train', action='store_true', help='If used, run the script with training mode.')
    parser.add_argument('--test', action='store_true', help='If used, run the script with test mode.')
    parser.add_argument('--polar-transform', type=str, default=None, help='Polar transformation. Should be one of linearpolar/logpolar.')
    parser.add_argument('--augmentation', type=str, default=None, help='Training data augmentation. Should be one of rot/trans/rottrans.')
    parser.add_argument('--data-dir', type=str, default='./data', help='Directory path to save datasets.')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size used in training.')
    parser.add_argument('--num-epochs', type=int, default=9999999, help='Number of maximum epochs.')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate.')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset. Should be one of mnist/svhn/cifar10/cifar100')
    parser.add_argument('--redirect', action='store_true', help='If used, redirect stdout to log file in logs/ .')
    parser.add_argument('--early-stop-epochs', type=float, default=15, help='Epochs to wait until early stopping.')
    parser.add_argument('--test-while-training', action='store_true', help='If used with --train, run tests at every training epoch.')
    parser.add_argument('--test-data-dir', type=str, default=None, help='Optional directory for test data.')
    parser.add_argument('--model-save-path', type=str, default=None, help='Path to save the trained model.')
    parser.add_argument('--output-dir', type=str, help='Output directory for confusion matrices.')

    args = vars(parser.parse_args())

    fname =  utils.generate_fname(args['dataset'], args['model'], args['polar_transform'], args['augmentation'])
    print('configuration: ', args)

    if args['redirect']:
        sys.stdout = open('logs/' + fname + '.txt', 'w')
        print('configuration: ', args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ', device)
    print('{} devices available'.format(torch.cuda.device_count()))

    model = get_model(model=args['model'], dataset=args['dataset'])

    print('# Parameters: {:.1f}K'.format(
        sum([p.numel() for p in model.parameters()]) / 1000
    ))

    criterion = nn.CrossEntropyLoss()

    train_loader, validation_loader, test_loader = None, None, None
    # """Load data """
    # train_loader, validation_loader, test_loader = \
    #         load_data(dataset=args['dataset'], data_dir=args['data_dir'], batch_size=args['batch_size'])

    """Load data"""
    if args['test'] and args['test_data_dir'] is not None:
        print(f"⚡ Loading test data from {args['test_data_dir']} ...")
        _, _, test_loader = load_data(dataset=args['dataset'], data_dir=args['test_data_dir'], batch_size=args['batch_size'])
    else:
        try:
            train_loader, validation_loader, test_loader = load_data(
                dataset=args['dataset'], data_dir=args['data_dir'], batch_size=args['batch_size']
            )
        except Exception as e:
            print(f"❌ Error loading training/validation data: {e}")
            sys.exit(1)

    if train_loader:
        print(f"✔️ Train data: {len(train_loader.dataset)} samples")
    else:
        print("⚠️ Train data not loaded.")

    if validation_loader:
        print(f"✔️ Validation data: {len(validation_loader.dataset)} samples")
    else:
        print("⚠️ Validation data not loaded.")

    if test_loader:
        print(f"✔️ Test data: {len(test_loader.dataset)} samples")
    else:
        print("⚠️ Test data not loaded.")
        
    # print('{} Train data. {} Validation data. {} Test data.'.format(
    #     len(train_loader.dataset), len(validation_loader.dataset), len(test_loader.dataset)
    # ))

    """ Test-Only (Using saved .pt file) """

    if args['test']:
        print(f"===> Testing model from {args['model_path']}")
        if args['model_path'] is None:
            print('Error: Please provide --model-path for testing.')
            sys.exit(1)
        checkpoint = torch.load(args['model_path'], map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        print('Model loaded successfully.')
        # print('===> Testing {} with rotated dataset begin'.format(fname))
        # checkpoint = torch.load('saves/' + fname + '.pt')
        # model.load_state_dict(checkpoint['state_dict'])
        # model.to(device)
        # output_dir = args.output_dir if args.output_dir else f"./logs/{args.train_set}_test_on_{args.test_set}"
        output_dir = args['output_dir'] if 'output_dir' in args and args['output_dir'] else f"./logs/{args['train_set']}_test_on_{args['test_set']}"

        os.makedirs(output_dir, exist_ok=True)
        test_loss, test_accuracy = test(model, device, criterion, test_loader, args, args['output_dir'])

        # test_loss, test_accuracy = test(model, device, criterion, test_loader, args,  output_dir, args.train_set, args.test_set)
        sys.exit(0)


    model.to(device)

    print('===> Training {} begin'.format(fname))

    optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=0.9, weight_decay=1e-5) #
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=10, cooldown=5, min_lr=1e-6, verbose=True)

    max_acc, last_saved = 0, 0

    if args['train']:
        for epoch in range(args['num_epochs']):

            start_time = time.time()

            train_loss = train(model, device, optimizer, criterion, train_loader, epoch, args)
            validation_loss, accuracy = validate(model, device, criterion, validation_loader, epoch, args)
            lr_scheduler.step(validation_loss)

            if args['test_while_training']:
                test_loss, test_accuracy = test(model, device, criterion, test_loader, args)
    

            if accuracy > max_acc:
                last_saved, max_acc = epoch, accuracy
                # save_path = f'saves/{fname}{args["model_save_path"]}.pt' if args['model_save_path'] else f'saves/{fname}.pt'
                save_path = args['model_save_path'] if args['model_save_path'] else f'saves/{fname}.pt'
                torch.save({
                    'state_dict': model.state_dict(),
                    'acc': max_acc,
                    'epoch': epoch,
                }, save_path)
                print(f"💾 Model saved at: {save_path}")


                # print('Saving model checkpoint to saves/{}.pt'.format(fname))
                

                # torch.save({
                #     'state_dict': model.state_dict(),
                #     'acc': max_acc,
                #     'epoch': epoch,
                # }, 'saves/' + fname + '.pt')

            """Elapsed time per epoch"""
            end_time = time.time()
            print('Elapsed time: {:.1f} sec'.format(end_time - start_time))

            """
            Check for ealry stop
            Look at recent 15 epochs
            If no significant improvment, terminate
            """
            if epoch >= 20 and epoch - last_saved > args['early_stop_epochs']:
                break

    print('Training Done!')


if __name__ == '__main__':
    main()
