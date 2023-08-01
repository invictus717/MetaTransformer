import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch
import itertools

class _UniNorm(Module):

    def __init__(self, num_features, dataset_from_flag=1, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, voxel_coord=False):
        super(_UniNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.voxel_coord = voxel_coord
        self.dataset_from_flag = dataset_from_flag
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean_source', torch.zeros(num_features))
            self.register_buffer('running_mean_target', torch.zeros(num_features))
            self.register_buffer('running_var_source', torch.ones(num_features))
            self.register_buffer('running_var_target', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean_source', None)
            self.register_parameter('running_mean_target', None)
            self.register_parameter('running_var_source', None)
            self.register_parameter('running_var_target', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.track_running_stats:
            self.running_mean_source.zero_()
            self.running_mean_target.zero_()
            self.running_var_source.fill_(1)
            self.running_var_target.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        return NotImplemented

    def _load_from_state_dict_from_pretrained_model(self, state_dict, prefix, metadata, strict, missing_keys, unexpected_keys, error_msgs):
        r"""Copies parameters and buffers from :attr:`state_dict` into only
        this module, but not its descendants. This is called on every submodule
        in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr`metadata`.
        For state dicts without meta data, :attr`metadata` is empty.
        Subclasses can achieve class-specific backward compatible loading using
        the version number at `metadata.get("version", None)`.

        .. note::
            :attr:`state_dict` is not the same object as the input
            :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
            it can be modified.

        Arguments:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module
            metadata (dict): a dict containing the metadata for this moodule.
                See
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (list of str): if ``strict=False``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=False``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`
        """
        local_name_params = itertools.chain(self._parameters.items(), self._buffers.items())
        local_state = {k: v.data for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if 'source' in key or 'target' in key:
                key = key[:-7]
                print(key)
            if key in state_dict:
                input_param = state_dict[key]
                if input_param.shape != param.shape:
                    # local shape should match the one in checkpoint
                    error_msgs.append('size mismatch for {}: copying a param of {} from checkpoint, '
                                      'where the shape is {} in current model.'
                                      .format(key, param.shape, input_param.shape))
                    continue
                if isinstance(input_param, Parameter):
                    # backwards compatibility for serialized parameters
                    input_param = input_param.data
                try:
                    param.copy_(input_param)
                except Exception:
                    error_msgs.append('While copying the parameter named "{}", '
                                      'whose dimensions in the model are {} and '
                                      'whose dimensions in the checkpoint are {}.'
                                      .format(key, param.size(), input_param.size()))
            elif strict:
                missing_keys.append(key)



    def forward(self, input):
        self._check_input_dim(input)
        if self.training :  ## train mode

            ## Split the input into the source and target batches
            ## and calculate the corresponding variances
            batch_size = input.size()[0] // 2
            input_source = input[:batch_size]
            input_target = input[batch_size:]

            ## In order to remap the rescaled source or target features into 
            ## the shared space, we use the shared self.weight and self.bias
            z_source = F.batch_norm(
                input_source, self.running_mean_source, self.running_var_source, self.weight, self.bias,
                self.training or not self.track_running_stats, self.momentum, self.eps)

            z_target = F.batch_norm(
                input_target, self.running_mean_target, self.running_var_target, self.weight, self.bias,
                self.training or not self.track_running_stats, self.momentum, self.eps)
            z = torch.cat((z_source, z_target), dim=0)

            # In order to address different dims
            if input.dim() == 4:    ## UniNorm2d
                input_source = input_source.permute(0,2,3,1).contiguous().view(-1,self.num_features)
                input_target = input_target.permute(0,2,3,1).contiguous().view(-1,self.num_features)

            cur_mean_source = torch.mean(input_source, dim=0)
            cur_var_source = torch.var(input_source,dim=0)
            cur_mean_target = torch.mean(input_target, dim=0)
            cur_var_target = torch.var(input_target, dim=0)

            ## Obtain the channel-wise transferability
            ## Assume that source_one and source_two features have different transferability along with channel dimension
            ## Global Statistic-level channel-wise transferability
            dis = torch.abs(cur_mean_source / torch.sqrt(cur_var_source + self.eps) -
                            cur_mean_target / torch.sqrt(cur_var_target + self.eps))

            ## Convert the channel-wise transferability into the probability distribution
            prob = 1.0 / (1.0 + dis)
            alpha = self.num_features * prob / sum(prob)

            # # Calculate the Cov Matrix
            # # Cov Matrix
            # if input_source.shape[0] == input_target.shape[0]:
            #     cov_matrix_src_tar = torch.matmul((input_source - cur_mean_source).T, (input_target - cur_mean_target)) / (input_source.shape[0]-1)
            
            #     # cov_vector = self.vote_cov(cov_matrix_src_tar)
            #     cov_vector = torch.diag(cov_matrix_src_tar)
            #     cov_vector = cov_vector.view(-1)
            #     alpha_cov = self.num_features * cov_vector / sum(cov_vector)
            #     alpha = (alpha + alpha_cov) / 2
            
            if input.dim() == 2:
                alpha = alpha.view(1, self.num_features)
            elif input.dim() == 4:
                alpha = alpha.view(1, self.num_features, 1, 1)

            ## Attention
            return z * (1 + alpha.detach())


        else:  ##test mode
            # for testing using multiple datasets in parallel, 
            # we need to split the input into the source and target batches
            # and calculate the corresponding variances
            batch_size = input.size()[0] // 2
            input_source = input[:batch_size]
            input_target = input[batch_size:]
            assert len(input_source) != 0 
            assert len(input_target) != 0

            # for source domain scaling:
            z_source = F.batch_norm(
                input_source, self.running_mean_source, self.running_var_source, self.weight, self.bias,
                self.training or not self.track_running_stats, self.momentum, self.eps)

            # for target domain scaling:
            z_target = F.batch_norm(
                input_target, self.running_mean_target, self.running_var_target, self.weight, self.bias,
                self.training or not self.track_running_stats, self.momentum, self.eps)

            dis = torch.abs(self.running_mean_source / torch.sqrt(self.running_var_source + self.eps)
                            - self.running_mean_target / torch.sqrt(self.running_var_target + self.eps))
            prob = 1.0 / (1.0 + dis)
            alpha = self.num_features * prob / sum(prob)

            z = torch.cat((z_source, z_target), dim=0)

            if input.dim() == 2:
                alpha = alpha.view(1, self.num_features)
            elif input.dim() == 4:
                alpha = alpha.view(1, self.num_features, 1, 1)
            return z * (1 + alpha.detach())

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


class UniNorm1d(_UniNorm):
    r"""Applies Batch Normalization over a 2D or 3D input (a mini-batch of 1D
    inputs with optional additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`_ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size).

    By default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momemtum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, L)` slices, it's common terminology to call this Temporal Batch Normalization.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, L)` or :math:`L` from input of size :math:`(N, L)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm1d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm1d(100, affine=False)
        >>> input = torch.randn(20, 100)
        >>> output = m(input)

    .. _`Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`:
        https://arxiv.org/abs/1502.03167
    """

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))


class UniNorm2d(_UniNorm):
    r"""Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`_ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size).

    By default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momemtum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial Batch Normalization.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm2d(100, affine=False)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> output = m(input)

    .. _`Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`:
        https://arxiv.org/abs/1502.03167
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class UniNorm3d(_UniNorm):
    r"""Applies Batch Normalization over a 5D input (a mini-batch of 3D inputs
    with additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`_ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size).

    By default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momemtum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, D, H, W)` slices, it's common terminology to call this Volumetric Batch Normalization
    or Spatio-temporal Batch Normalization.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, D, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm3d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm3d(100, affine=False)
        >>> input = torch.randn(20, 100, 35, 45, 10)
        >>> output = m(input)

    .. _`Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`:
        https://arxiv.org/abs/1502.03167
    """

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))