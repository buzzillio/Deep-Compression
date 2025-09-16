import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

from net.models import LeNet
from net.quantization import apply_weight_sharing
import util

os.makedirs('saves', exist_ok=True)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST pruning from deep compression paper')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 50)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--log', type=str, default='log.txt',
                    help='log file name')
parser.add_argument('--sensitivity', type=float, default=2,
                    help="sensitivity value that is multiplied to layer's std in order to get threshold value")
parser.add_argument('--pruning-method', type=str, default='tfidf', choices=['std', 'tfidf'],
                    help='method used for pruning (default: tfidf)')
parser.add_argument('--tfidf-activation-threshold', type=float, default=0.05,
                    help='activation threshold used when computing document frequency for TF-IDF pruning')
parser.add_argument('--tfidf-idf-smooth', type=float, default=1.0,
                    help='smoothing value added to the IDF numerator and denominator')
parser.add_argument('--tfidf-idf-add', type=float, default=1.0,
                    help='constant added to the IDF term before applying the power')
parser.add_argument('--tfidf-idf-power', type=float, default=1.0,
                    help='exponent applied to the IDF term during TF-IDF pruning')
parser.add_argument('--tfidf-tf-power', type=float, default=1.0,
                    help='exponent applied to the mean activation (TF) term during TF-IDF pruning')
parser.add_argument('--tfidf-weight-power', type=float, default=1.0,
                    help='exponent applied to weight magnitudes during TF-IDF pruning')
parser.add_argument('--tfidf-global-threshold', action='store_true', default=False,
                    help='if set, compute a single TF-IDF threshold across all prunable layers')
parser.add_argument('--tfidf-percentile', type=float, default=None,
                    help='optional percentile (0-100) for TF-IDF pruning; overrides sensitivity when set')
parser.add_argument('--tfidf-max-batches', type=int, default=None,
                    help='maximum number of batches to use when collecting TF-IDF activation statistics')
args = parser.parse_args()

# Control Seed
torch.manual_seed(args.seed)

# Select Device
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')
if use_cuda:
    print("Using CUDA!")
    torch.cuda.manual_seed(args.seed)
else:
    print('Not using CUDA!!!')

# Loader
kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)


# Define which model to use
model = LeNet(mask=True).to(device)

print(model)
util.print_model_parameters(model)

# NOTE : `weight_decay` term denotes L2 regularization loss term
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
initial_optimizer_state_dict = optimizer.state_dict()


def collect_activation_statistics(model, data_loader, device, activation_threshold=0.05, max_batches=None):
    """Collect mean absolute activations and document frequency per masked layer."""
    activation_threshold = max(0.0, activation_threshold)
    stats = {}

    handles = []
    for name, module in model.named_modules():
        if hasattr(module, 'mask'):
            def make_hook(layer_name):
                def hook(_module, inputs, _output):
                    if not inputs:
                        return
                    features = inputs[0]
                    if features is None:
                        return
                    if features.dim() == 1:
                        features = features.unsqueeze(0)
                    batch_size = features.size(0)
                    if batch_size == 0:
                        return
                    flattened = features.detach().reshape(batch_size, -1).abs().to(dtype=torch.double, device='cpu')
                    present = (flattened > activation_threshold).to(dtype=torch.double)

                    layer_stats = stats.setdefault(layer_name, {
                        'sum_abs_activation': torch.zeros(flattened.size(1), dtype=torch.double),
                        'doc_freq': torch.zeros(flattened.size(1), dtype=torch.double),
                        'sample_count': 0,
                    })
                    layer_stats['sum_abs_activation'] += flattened.sum(dim=0)
                    layer_stats['doc_freq'] += present.sum(dim=0)
                    layer_stats['sample_count'] += batch_size

                return hook

            handles.append(module.register_forward_hook(make_hook(name)))

    if not handles:
        print('No masked layers found when collecting activation statistics.')
        return {}

    was_training = model.training
    model.eval()
    processed_samples = 0
    with torch.no_grad():
        for batch_idx, (data, _target) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            data = data.to(device)
            processed_samples += data.size(0)
            model(data)

    for handle in handles:
        handle.remove()

    if was_training:
        model.train()

    for layer_name, layer_stats in stats.items():
        count = layer_stats['sample_count']
        if count > 0:
            layer_stats['mean_abs_activation'] = layer_stats['sum_abs_activation'] / count
        else:
            layer_stats['mean_abs_activation'] = torch.zeros_like(layer_stats['sum_abs_activation'])
        layer_stats['doc_freq'] = layer_stats['doc_freq'].clamp_(min=0.0, max=float(count))
        del layer_stats['sum_abs_activation']

    print(f'Collected activation statistics from {processed_samples} samples for TF-IDF pruning.')
    return stats

def train(epochs):
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (data, target) in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()

            # zero-out all the gradients corresponding to the pruned connections
            for name, p in model.named_parameters():
                if 'mask' in name:
                    continue
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor==0, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)

            optimizer.step()
            if batch_idx % args.log_interval == 0:
                done = batch_idx * len(data)
                percentage = 100. * batch_idx / len(train_loader)
                pbar.set_description(f'Train Epoch: {epoch} [{done:5}/{len(train_loader.dataset)} ({percentage:3.0f}%)]  Loss: {loss.item():.6f}')


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy


# Initial training
print("--- Initial training ---")
train(args.epochs)
accuracy = test()
util.log(args.log, f"initial_accuracy {accuracy}")
torch.save(model, f"saves/initial_model.ptmodel")
print("--- Before pruning ---")
util.print_nonzeros(model)

# Pruning
if args.pruning_method == 'tfidf':
    print('--- Collecting statistics for TF-IDF pruning ---')
    activation_stats = collect_activation_statistics(
        model,
        train_loader,
        device,
        activation_threshold=args.tfidf_activation_threshold,
        max_batches=args.tfidf_max_batches,
    )
    model.prune_by_tfidf(
        activation_stats,
        sensitivity=args.sensitivity,
        percentile=args.tfidf_percentile,
        global_threshold=args.tfidf_global_threshold,
        idf_smooth=args.tfidf_idf_smooth,
        idf_add=args.tfidf_idf_add,
        idf_power=args.tfidf_idf_power,
        tf_power=args.tfidf_tf_power,
        weight_power=args.tfidf_weight_power,
    )
else:
    model.prune_by_std(args.sensitivity)
accuracy = test()
util.log(args.log, f"accuracy_after_pruning {accuracy}")
print("--- After pruning ---")
util.print_nonzeros(model)

# Retrain
print("--- Retraining ---")
optimizer.load_state_dict(initial_optimizer_state_dict) # Reset the optimizer
train(args.epochs)
torch.save(model, f"saves/model_after_retraining.ptmodel")
accuracy = test()
util.log(args.log, f"accuracy_after_retraining {accuracy}")

print("--- After Retraining ---")
util.print_nonzeros(model)
