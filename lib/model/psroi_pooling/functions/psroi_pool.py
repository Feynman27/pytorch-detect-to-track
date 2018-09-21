import torch
from torch.autograd import Function
from .._ext import psroi_pooling


class PSRoIPoolFunction(Function):
    def __init__(ctx, pooled_height, pooled_width, spatial_scale, group_size, output_dim):
        ctx.pooled_width = int(pooled_width)
        ctx.pooled_height = int(pooled_height)
        ctx.spatial_scale = float(spatial_scale)
        ctx.group_size = int(group_size)
        ctx.output_dim = int(output_dim)
        ctx.output = None
        ctx.mappingchannel = None
        ctx.rois = None
        ctx.feature_size = None

    def forward(ctx, features, rois):
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size()[0]
        output = torch.zeros(num_rois, ctx.output_dim, ctx.pooled_height, ctx.pooled_width)
        mappingchannel = torch.IntTensor(num_rois, ctx.output_dim, ctx.pooled_height, ctx.pooled_width).zero_()
        output = output.cuda()
        mappingchannel = mappingchannel.cuda()
        psroi_pooling.psroi_pooling_forward_cuda(ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale,
                                                 ctx.group_size, ctx.output_dim,
                                                 features, rois, output, mappingchannel)
        ctx.output = output
        ctx.mappingchannel = mappingchannel
        ctx.rois = rois
        ctx.feature_size = features.size()

        return output

    def backward(ctx, grad_output):
        assert(ctx.feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_height, data_width = ctx.feature_size

        grad_input = torch.zeros(batch_size, num_channels, data_height, data_width).cuda()

        psroi_pooling.psroi_pooling_backward_cuda(ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale,
                                                  ctx.output_dim, grad_output,
                                                  ctx.rois, grad_input, ctx.mappingchannel)
        return grad_input, None

#class PSRoIPool(torch.nn.Module):
#    def __init__(self, pooled_height, pooled_width, spatial_scale, group_size, output_dim):
#        super(PSRoIPool, self).__init__()
#
#        self.pooled_width = int(pooled_width)
#        self.pooled_height = int(pooled_height)
#        self.spatial_scale = float(spatial_scale)
#        self.group_size = int(group_size)
#        self.output_dim = int(output_dim)

#    def forward(self, features, rois):
#        return PSRoIPoolFunction(self.pooled_height, self.pooled_width, self.spatial_scale, self.group_size, self.output_dim)(features, rois)
