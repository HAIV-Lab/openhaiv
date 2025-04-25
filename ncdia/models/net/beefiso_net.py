import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet.cifar_resnet import resnet32
import copy

from ncdia.utils import MODELS, Configs


class SimpleLinear(nn.Module):
    '''
    Reference:
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
    '''
    def __init__(self, in_features, out_features, bias=True):
        super(SimpleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')
        nn.init.constant_(self.bias, 0)

    def forward(self, input):
        return {'logits': F.linear(input, self.weight, self.bias)}
    
class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x , bias=True):
        ret_x = x.clone()
        ret_x = (self.alpha+1) * x # + self.beta
        if bias:
            ret_x = ret_x + self.beta
        return ret_x

    def get_params(self):
        return (self.alpha.item(), self.beta.item())
        
@MODELS.register
class BEEFISONet(nn.Module):
    def __init__(
            self,
            network: Configs,
            base_classes,
            num_classes,
            att_classes,
    ) -> None:
        super(BEEFISONet, self).__init__()
        self.convnets = nn.ModuleList()
        self.out_dim = None
        self.old_fc = None
        self.new_fc = None
        self.task_sizes = []
        self.forward_prototypes = None
        self.backward_prototypes = None
        self.biases = nn.ModuleList()

        self.args = network.cfg
        self.args['pretrained'] = True
        self.args['num_classes'] = 256
        self.args['type'] = 'resnet18'

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x):
        features = [convnet(x) for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        # print(x.shape)  [bs, 16, 224, 224]
        features = [convnet(x)['features'] for convnet in self.convnets]
        # print(features[0].shape)
        features = torch.cat(features, 1)
        
        # print(self.old_fc)
        if self.old_fc is None:
            fc = self.new_fc
            out = fc(features)
        else:
            '''
            merge the weights
            '''
            # print(self.task_sizes)
            new_task_size = self.task_sizes[-1]
            # print(self.feature_dim,self.out_dim,new_task_size)
            fc_weight = torch.cat([self.old_fc.weight,torch.zeros((new_task_size,self.feature_dim-self.out_dim)).cuda()],dim=0)             
            new_fc_weight = self.new_fc.weight
            new_fc_bias = self.new_fc.bias
            for i in range(len(self.task_sizes)-2,-1,-1):
                new_fc_weight = torch.cat([*[self.biases[i](self.backward_prototypes.weight[i].unsqueeze(0),bias=False) for _ in range(self.task_sizes[i])],new_fc_weight],dim=0)
                new_fc_bias = torch.cat([*[self.biases[i](self.backward_prototypes.bias[i].unsqueeze(0),bias=True) for _ in range(self.task_sizes[i])], new_fc_bias])
            fc_weight = torch.cat([fc_weight,new_fc_weight],dim=1)
            fc_bias = torch.cat([self.old_fc.bias,torch.zeros(new_task_size).cuda()])
            fc_bias=+new_fc_bias
            logits = features@fc_weight.permute(1,0)+fc_bias
            out = {"logits":logits}        

            new_fc_weight = self.new_fc.weight
            new_fc_bias = self.new_fc.bias
            for i in range(len(self.task_sizes)-2,-1,-1):
                new_fc_weight = torch.cat([self.backward_prototypes.weight[i].unsqueeze(0),new_fc_weight],dim=0)
                new_fc_bias = torch.cat([self.backward_prototypes.bias[i].unsqueeze(0), new_fc_bias])
            out["train_logits"] = features[:,-self.out_dim:]@new_fc_weight.permute(1,0)+new_fc_bias 
        out.update({"eval_logits": out["logits"],"energy_logits":self.forward_prototypes(features[:,-self.out_dim:])["logits"]})
        return out

    def update_fc_before(self, nb_classes):
        print(nb_classes)
        print(self.task_sizes)
        new_task_size = nb_classes - sum(self.task_sizes)
        self.biases = nn.ModuleList([BiasLayer() for i in range(len(self.task_sizes))])
        # self.args['type'] = 'resnet18'
        # model = MODELS.build(self.args)
        model = resnet32()
        # print(model)
        if not hasattr(model, 'out_dim'):
            # 添加 out_dim 属性（假设是 ResNet）
            model.out_dim = model.fc.in_features
        # print(model.out_dim)
        self.convnets.append(model)
        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim
        if self.new_fc is not None:
            self.fe_fc = self.generate_fc(self.out_dim, nb_classes)
            self.backward_prototypes = self.generate_fc(self.out_dim,len(self.task_sizes))
            self.convnets[-1].load_state_dict(self.convnets[0].state_dict())
        self.forward_prototypes = self.generate_fc(self.out_dim, nb_classes)
        # print(self.out_dim,new_task_size) 64*20
        self.new_fc = self.generate_fc(self.out_dim,new_task_size)
        self.task_sizes.append(new_task_size)
    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc
    
    def update_fc_after(self):
        if self.old_fc is not None:
            old_fc = self.generate_fc(self.feature_dim, sum(self.task_sizes))
            new_task_size = self.task_sizes[-1]
            old_fc.weight.data = torch.cat([self.old_fc.weight.data,torch.zeros((new_task_size,self.feature_dim-self.out_dim)).cuda()],dim=0)             
            new_fc_weight = self.new_fc.weight.data
            new_fc_bias = self.new_fc.bias.data
            for i in range(len(self.task_sizes)-2,-1,-1):
                new_fc_weight = torch.cat([*[self.biases[i](self.backward_prototypes.weight.data[i].unsqueeze(0),bias=False) for _ in range(self.task_sizes[i])], new_fc_weight],dim=0)
                new_fc_bias = torch.cat([*[self.biases[i](self.backward_prototypes.bias.data[i].unsqueeze(0),bias=True) for _ in range(self.task_sizes[i])], new_fc_bias])
            old_fc.weight.data = torch.cat([old_fc.weight.data,new_fc_weight],dim=1)
            old_fc.bias.data = torch.cat([self.old_fc.bias.data,torch.zeros(new_task_size).cuda()])
            old_fc.bias.data+=new_fc_bias
            self.old_fc = old_fc
        else:
            self.old_fc  = self.new_fc

    def copy(self):
        return copy.deepcopy(self)

    def copy_fc(self, fc):
        weight = copy.deepcopy(fc.weight.data)
        bias = copy.deepcopy(fc.bias.data)
        n, m = weight.shape[0], weight.shape[1]
        self.fc.weight.data[:n, :m] = weight
        self.fc.bias.data[:n] = bias

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, old, increment, value):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew * (value ** (old / increment))
        # logging.info("align weights, gamma = {} ".format(gamma))
        self.fc.weight.data[-increment:, :] *= gamma