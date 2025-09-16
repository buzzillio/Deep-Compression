import math

import numpy as np
import torch
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

class PruningModule(Module):
    def prune_by_percentile(self, q=5.0, **kwargs):
        """
        Note:
             The pruning percentile is based on all layer's parameters concatenated
        Args:
            q (float): percentile in float
            **kwargs: may contain `cuda`
        """
        # Calculate percentile value
        alive_parameters = []
        for name, p in self.named_parameters():
            # We do not prune bias term
            if 'bias' in name or 'mask' in name:
                continue
            tensor = p.data.cpu().numpy()
            alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
            alive_parameters.append(alive)

        all_alives = np.concatenate(alive_parameters)
        percentile_value = np.percentile(abs(all_alives), q)
        print(f'Pruning with threshold : {percentile_value}')

        # Prune the weights and mask
        # Note that module here is the layer
        # ex) fc1, fc2, fc3
        for name, module in self.named_modules():
            if name in ['fc1', 'fc2', 'fc3']:
                module.prune(threshold=percentile_value)

    def prune_by_std(self, s=0.25):
        """
        Note that `s` is a quality parameter / sensitivity value according to the paper.
        According to Song Han's previous paper (Learning both Weights and Connections for Efficient Neural Networks),
        'The pruning threshold is chosen as a quality parameter multiplied by the standard deviation of a layer’s weights'

        I tried multiple values and empirically, 0.25 matches the paper's compression rate and number of parameters.
        Note : In the paper, the authors used different sensitivity values for different layers.
        """
        for name, module in self.named_modules():
            if name in ['fc1', 'fc2', 'fc3']:
                threshold = np.std(module.weight.data.cpu().numpy()) * s
                print(f'Pruning with threshold : {threshold} for layer {name}')
                module.prune(threshold)

    def prune_by_tfidf(
        self,
        activation_stats,
        sensitivity=1.0,
        percentile=None,
        global_threshold=False,
        idf_smooth=1.0,
        idf_add=1.0,
        idf_power=1.0,
        tf_power=1.0,
        weight_power=1.0,
    ):
        """Prune connections using a TF-IDF inspired score.

        Args:
            activation_stats (dict): Statistics collected from a dataset. Each key is a
                module name and each value is a dict containing ``mean_abs_activation``,
                ``doc_freq`` and ``sample_count`` tensors.
            sensitivity (float): Multiplier applied to the standard deviation of the
                scores when ``percentile`` is not provided. Higher values keep more
                connections.
            percentile (float, optional): If provided, prune connections with scores
                below the given percentile (0-100). Overrides ``sensitivity``.
            global_threshold (bool): If ``True`` compute a single threshold across all
                prunable layers, otherwise compute a per-layer threshold.
            idf_smooth (float): Smoothing term added to the numerator and denominator
                of the IDF computation.
            idf_add (float): Constant added to the IDF term before applying
                ``idf_power``. Defaults to the classic ``+1`` used in TF-IDF.
            idf_power (float): Exponent applied to the IDF term.
            tf_power (float): Exponent applied to the TF (mean absolute activation)
                term.
            weight_power (float): Exponent applied to the absolute weight magnitude.
        """

        if percentile is not None and not (0.0 <= percentile <= 100.0):
            raise ValueError('percentile must be between 0 and 100')

        eps = 1e-12
        layer_data = []
        global_scores = []

        for name, module in self.named_modules():
            if not hasattr(module, 'mask'):
                continue

            stats = activation_stats.get(name)
            if not stats:
                continue

            sample_count = stats.get('sample_count', 0)
            if sample_count == 0:
                continue

            mean_abs_activation = stats['mean_abs_activation'].to(torch.double)
            doc_freq = stats['doc_freq'].to(torch.double)
            weight = module.weight.detach().to(torch.device('cpu'), dtype=torch.double)
            mask = module.mask.detach().to(torch.device('cpu'), dtype=torch.double)

            if weight.numel() == 0:
                continue

            weight_component = weight.abs().pow(weight_power)
            tf_component = mean_abs_activation.clamp(min=0.0).pow(tf_power)
            smooth = idf_smooth if idf_smooth > 0 else 0.0
            numerator = sample_count + smooth + eps
            denominator = doc_freq + smooth + eps
            idf_component = torch.log(numerator / denominator)
            if idf_add != 0.0:
                idf_component = idf_component + idf_add
            idf_component = idf_component.clamp(min=0.0).pow(idf_power)

            scores = weight_component * tf_component.unsqueeze(0) * idf_component.unsqueeze(0)
            scores = scores * mask

            alive_mask = mask > 0
            if not torch.any(alive_mask):
                continue

            alive_scores = scores[alive_mask]
            layer_record = {
                'name': name,
                'module': module,
                'scores': scores,
                'alive_scores': alive_scores,
                'mask': mask,
            }
            layer_data.append(layer_record)

            if global_threshold:
                global_scores.append(alive_scores)

        if not layer_data:
            print('No layers eligible for TF-IDF pruning. Skipping.')
            return

        if global_threshold:
            all_scores = torch.cat(global_scores) if global_scores else torch.tensor([], dtype=torch.double)
            if all_scores.numel() == 0:
                print('No alive scores found for global TF-IDF pruning. Skipping.')
                return
            if percentile is not None:
                threshold_value = float(np.percentile(all_scores.numpy(), percentile))
                print(f'Global TF-IDF pruning threshold (percentile {percentile}): {threshold_value}')
            else:
                score_std = all_scores.std(unbiased=False).item()
                threshold_value = score_std * sensitivity
                print(f'Global TF-IDF pruning threshold (std {score_std} * sensitivity {sensitivity}): {threshold_value}')

            for record in layer_data:
                name = record['name']
                module = record['module']
                scores = record['scores']
                original_mask = record['mask']
                prunable = original_mask > 0
                pruned = int(torch.sum(prunable & (scores < threshold_value)).item())
                total = int(torch.sum(prunable).item())
                print(f'Layer {name}: pruning {pruned}/{total} connections using global TF-IDF threshold {threshold_value}')
                module.prune_with_scores(scores, threshold_value)
            return

        for record in layer_data:
            name = record['name']
            module = record['module']
            scores = record['scores']
            alive_scores = record['alive_scores']
            original_mask = record['mask']
            prunable = original_mask > 0

            if percentile is not None:
                threshold_value = float(np.percentile(alive_scores.numpy(), percentile))
                print(f'Layer {name}: TF-IDF pruning threshold (percentile {percentile}): {threshold_value}')
            else:
                score_std = alive_scores.std(unbiased=False).item()
                threshold_value = score_std * sensitivity
                print(f'Layer {name}: TF-IDF pruning threshold (std {score_std} * sensitivity {sensitivity}): {threshold_value}')

            pruned = int(torch.sum(prunable & (scores < threshold_value)).item())
            total = int(torch.sum(prunable).item())
            print(f'Layer {name}: pruning {pruned}/{total} connections using TF-IDF scores')
            module.prune_with_scores(scores, threshold_value)


class MaskedLinear(Module):
    r"""Applies a masked linear transformation to the incoming data: :math:`y = (A * M)x + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            (out_features x in_features)
        bias:   the learnable bias of the module of shape (out_features)
        mask: the unlearnable mask for the weight.
            It has the same shape as weight (out_features x in_features)

    """
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        # Initialize the mask with 1
        self.mask = Parameter(torch.ones([out_features, in_features]), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.linear(input, self.weight * self.mask, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'

    def prune(self, threshold):
        mask = self.mask.data.clone()
        zero_mask = torch.zeros_like(mask)
        new_mask = torch.where(self.weight.data.abs() < threshold, zero_mask, mask)
        self.apply_new_mask(new_mask)

    def prune_with_scores(self, scores, threshold):
        mask = self.mask.data.clone()
        zero_mask = torch.zeros_like(mask)
        score_tensor = scores.to(mask.device, dtype=mask.dtype)
        new_mask = torch.where((score_tensor < threshold) & (mask > 0), zero_mask, mask)
        self.apply_new_mask(new_mask)

    def apply_new_mask(self, new_mask):
        new_mask = new_mask.to(self.mask.device, dtype=self.mask.dtype)
        self.mask.data = new_mask
        self.weight.data = self.weight.data.to(self.weight.device) * new_mask.to(self.weight.device)




