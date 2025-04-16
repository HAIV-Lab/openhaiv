import copy
import torch
import torch.nn as nn
from ncdia.utils import MODELS, Configs

def norm(x):
    norm = torch.norm(x, p=2, dim=1)
    x = x / (norm.expand(1, -1).t() + .0001)
    return x

@MODELS.register
class CosineDeconf(nn.Module):
    def __init__(self, in_features, num_classes):
        super(CosineDeconf, self).__init__()

        self.h = nn.Linear(in_features, num_classes, bias=False)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity='relu')

    def forward(self, x):
        x = norm(x)
        w = norm(self.h.weight)

        ret = (torch.matmul(x, w.T))
        return ret

@MODELS.register
class EuclideanDeconf(nn.Module):
    def __init__(self, in_features, num_classes):
        super(EuclideanDeconf, self).__init__()

        self.h = nn.Linear(in_features, num_classes, bias=False)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity='relu')

    def forward(self, x):

        # size: (batch, latent, 1)
        x = x.unsqueeze(2)

        # size: (1, latent, num_classes)
        h = self.h.weight.T.unsqueeze(0)
        ret = -((x - h).pow(2)).mean(1)
        return ret

@MODELS.register
class InnerDeconf(nn.Module):
    def __init__(self, in_features, num_classes):
        super(InnerDeconf, self).__init__()

        self.h = nn.Linear(in_features, num_classes)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity='relu')
        self.h.bias.data = torch.zeros(size=self.h.bias.size())

    def forward(self, x):
        return self.h(x)

@MODELS.register
class GodinNet(nn.Module):
    def __init__(self,
                 network: Configs,
                 checkpoint: str = None,
                 feature_size: int = 2048,
                 num_classes: int = 94,
                 similarity_measure='cosine',
                 loss: str = 'CrossEntropyLoss', 
                 **kwargs)-> None:
        super(GodinNet, self).__init__()

        h_dict = {
            'cosine': CosineDeconf,
            'inner': InnerDeconf,
            'euclid': EuclideanDeconf
        }

        self.num_classes = num_classes
        self.args = network.cfg
        self.args['pretrained'] = True
        num_classes_true = self.args['num_classes']
        self.args['num_classes'] = 1000
        self.network = MODELS.build(copy.deepcopy(self.args))
        self.network.fc = torch.nn.Linear(feature_size, num_classes_true)
        
        self.noise_magnitude =  0.0025
        self.backbone = self.network
            
        if hasattr(self.backbone, 'fc'):
            # remove fc otherwise ddp will
            # report unused params
            self.backbone.fc = nn.Identity()

        self.h = h_dict[similarity_measure](feature_size, self.num_classes)

        self.g = nn.Sequential(nn.Linear(feature_size, 1), nn.BatchNorm1d(1),
                               nn.Sigmoid())
        self.loss = loss
        self.softmax = nn.Softmax()
        
        self.input_std = [0.5, 0.5, 0.5]
        if checkpoint:
            key_dict = checkpoint.split(' ')
            checkpoint = key_dict[0]
            self.inference = key_dict[1]
            print('load_checkpoint')
            state_dict = torch.load(checkpoint)
            state_dict = {k.replace('network.', ''): v for k, v in state_dict['state_dict'].items()}
            self.backbone.load_state_dict(state_dict, strict=False)
            
            
    def backbone_forward(self, x):
        feature1 = self.backbone.relu(self.network.bn1(self.network.conv1(x)))
        feature1 = self.backbone.maxpool(feature1)
        feature2 = self.backbone.layer1(feature1)
        feature3 = self.backbone.layer2(feature2)
        feature4 = self.backbone.layer3(feature3)
        feature5 = self.backbone.layer4(feature4)
        feature5 = self.backbone.avgpool(feature5)
        feature = feature5.view(feature5.size(0), -1)
        logits_cls = self.backbone.fc(feature)

        # x = self.network(x)
        return logits_cls, feature
        
    def forward(self, x, inference=False, score_func='h'):
        x.requires_grad = True
        _, feature = self.backbone_forward(x)

        numerators = self.h(feature)

        denominators = self.g(feature)

        # calculate the logits results
        quotients = numerators / denominators

        # logits, numerators, and denominators
        if self.inference:
            if score_func == 'h':
                output =  numerators
            elif score_func == 'g':
                output =  denominators
            max_scores, _ = torch.max(output, dim=1)
            max_scores.backward(torch.ones(len(max_scores)).cuda())

            # Normalizing the gradient to binary in {0, 1}
            gradient = torch.ge(x.grad.detach(), 0)
            gradient = (gradient.float() - 0.5) * 2

            # Scaling values taken from original code
            gradient[:, 0] = (gradient[:, 0]) / self.input_std[0]
            gradient[:, 1] = (gradient[:, 1]) / self.input_std[1]
            gradient[:, 2] = (gradient[:, 2]) / self.input_std[2]

            # Adding small perturbations to images
            tempInputs = torch.add(x.detach(),
                                   gradient,
                                   alpha=self.noise_magnitude)

            # calculate score
            _, feature = self.backbone_forward(tempInputs)

            numerators_tmp = self.h(feature)

            denominators_tmp = self.g(feature)

            if score_func == 'h':
                output = numerators_tmp
            elif score_func == 'g':
                output =  denominators

            nnOutput = output.detach()
            nnOutput = nnOutput - nnOutput.max(dim=1, keepdims=True).values
            nnOutput = nnOutput.exp() / nnOutput.exp().sum(dim=1, keepdims=True)


            return nnOutput
            
        else:
            return quotients
            
    def get_features(self,x):
        feature1 = self.backbone.relu(self.network.bn1(self.network.conv1(x)))
        feature1 = self.backbone.maxpool(feature1)
        feature2 = self.backbone.layer1(feature1)
        feature3 = self.backbone.layer2(feature2)
        feature4 = self.backbone.layer3(feature3)
        feature5 = self.backbone.layer4(feature4)
        feature5 = self.backbone.avgpool(feature5)
        feature = feature5.view(feature5.size(0), -1)
        logits_cls = self.backbone.fc(feature)
        return feature