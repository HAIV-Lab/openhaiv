import torch
import torch.nn.functional as F

from ncdia.utils import ALGORITHMS
from ncdia.algorithms.base import BaseAlg
from ncdia.utils.losses import AngularPenaltySMLoss
from ncdia.dataloader.augmentations import fantasy
from ncdia.utils.metrics import accuracy
from .hooks import SAVCHook


@ALGORITHMS.register
class SAVC(BaseAlg):
    def __init__(self, trainer) -> None:
        super().__init__(trainer)
        self.args = trainer.cfg
        self.config = self.args
        # print(self.args)

        self._network = None
        self.loss = AngularPenaltySMLoss()
        self.hook = SAVCHook()
        trainer.register_hook(self.hook)

        if self.args.CIL.fantasy is not None:
            self.transform, self.num_trans = fantasy.__dict__[self.args.CIL.fantasy]()
        else:
            self.transform = None
            self.num_trans = 0
        
        session = trainer.session
        if session == 1:
            self._network = trainer.model
            self._network.eval()
            self._network.mode = self.args.CIL.new_mode
            trainer.train_loader.dataset.multi_train =False
            trainloader = trainer.train_loader
            tsfm = trainer.val_loader.dataset.transform
            trainloader.dataset.transform = tsfm
            class_list = list(range(self.args.CIL.base_class+ (session-1)*self.args.CIL.way, self.args.CIL.base_class + self.args.CIL.way * session))
            self._network.update_fc(trainloader, class_list, session)


    def train_step(self, trainer, data, label, *args, **kwargs):
        """
        base train for fact method
        Args:
            trainer: trainer object
            data: data in batch
            label: label in batch
            attribute: attribute in batch
            imgpath: imgpath in batch
        """
        session = trainer.session
        if session==0:
            self._network = trainer.model
            self._network.train()
            device = trainer.device
            b, c, h, w = data[1].shape
            original = data[0].to(device)
            data[1] = data[1].to(device)
            data[2] = data[2].to(device)
            label = label.to(device)

            if len(self.config.CIL.num_crops) > 1:
                    data_small = data[self.config.CIL.num_crops[0]+1].unsqueeze(1)
                    for j in range(1, self.config.CIL.num_crops[1]):
                        data_small = torch.cat((data_small, data[j+self.config.CIL.num_crops[0]+1].unsqueeze(1)), dim=1)
                    data_small = data_small.view(-1, c, self.config.CIL.size_crops[1], \
                                                self.config.CIL.size_crops[1]).cuda(non_blocking=True)
            else:
                data_small = None

            data_classify = self.transform(original)    
            data_query = self.transform(data[1])
            data_key = self.transform(data[2])
            data_small = self.transform(data_small)

            m = data_query.size()[0] // b
            joint_labels = torch.stack([label*m+ii for ii in range(m)], 1).view(-1)

            # ------  forward  ------- #
            joint_preds = self._network(im_cla=data_classify)  
            joint_preds = joint_preds[:, :self.config.CIL.base_class*m]
            joint_loss = F.cross_entropy(joint_preds, joint_labels)

            agg_preds = 0
            for i in range(m):
                agg_preds = agg_preds + joint_preds[i::m, i::m] / m

            loss = joint_loss
            loss.backward()


            # acc = self._accuracy(joint_labels, joint_preds)
            # print(joint_labels.shape)
            # print(joint_preds.shape)
            # input()
            acc = accuracy(joint_preds, joint_labels)[0]

            ret = {}
            ret['loss'] = loss
            ret['acc'] = acc
            return ret
        else:
            ret = {}

        return ret

    def val_step(self, trainer, data, label, *args, **kwargs):
        """Validation step for standard supervised learning.

        Args:
            trainer (object): Trainer object.
            data (torch.Tensor): Input data.
            label (torch.Tensor): Label data.
            args (tuple): Additional arguments.
            kwargs (dict): Additional keyword arguments.

        Returns:
            results (dict): Validation results. Contains the following:
                - "feature" (numpy.array): features in a batch
                - "logits" (numpy.array): logits in a batch
                - "predicts" (numpy.array): predicts in a batch
                - "confidence" (numpy.array): confidence in a batch
                - "label" (numpy.array): labels in a batch
                - "loss": Loss value.
                - "acc": Accuracy value.
        """
        self._network = trainer.model
        self._network.eval()
        device = trainer.device
        session = trainer.session
        if session==1:
            with torch.no_grad():
                data = data.cuda()
                labels = label.cuda()

                logits = self._network(data)
                logits_ = logits[:, :self.config.CIL.base_class]
                # _, pred = torch.max(logits_, dim=1)
                # acc = self._accuracy(labels, pred)
                acc = accuracy(logits_, labels)[0]
                loss = self.loss(logits_, labels)
                
                ret = {}
                ret['loss'] = loss.item()
                ret['acc'] = acc.item()
                return ret
        else:
            logit_list, pred_list, conf_list, label_list = [], [], [], []
            feature_list = []
            test_class = self.config.CIL.base_class + 1 * self.config.CIL.way
            with torch.no_grad():
                data = data.to(device)
                label = label.to(device)
                b = data.size()[0]
                if self.transform is not None:
                    data = self.transform(data)
                m = data.size()[0] // b
                joint_preds = self._network(data)
                feat = self._network.get_features(data)
                joint_preds = joint_preds[:, :test_class*m]
                
                agg_preds = 0
                agg_feat = feat.view(-1, m, feat.size(1)).mean(dim=1)
                for j in range(m):
                    agg_preds = agg_preds + joint_preds[j::m, j::m] / m
                
                # loss = F.cross_entropy(joint_preds, label)
                
                feature_list.append(agg_feat)
                logit_list.append(agg_preds)
                score = torch.softmax(agg_preds, dim=1)
                conf, pred = torch.max(score, dim=1)
                acc = self._accuracy(pred, label)
                

                pred_list.append(pred.cpu())
                conf_list.append(conf.cpu())
                label_list.append(label.cpu())

            feature_list = torch.cat(feature_list, dim=0).cpu().numpy()
            logit_list = torch.cat(logit_list, dim=0).cpu().numpy()
            pred_list = torch.cat(pred_list).numpy().astype(int)
            conf_list = torch.cat(conf_list).numpy()
            label_list = torch.cat(label_list).numpy().astype(int)
            # loss = F.cross_entropy(torch.from_numpy(label_list), torch.from_numpy(logit_list))
            ret = {}
            ret['acc']=acc
            # ret['loss']=loss

        
        return ret
    
    def replace_fc(self):
        session = self.trainer.session
        if not self.args.CIL.not_data_init and session == 0:
            train_loader = self.trainer.train_loader
            val_loader = self.trainer.val_loader
            train_loader.dataset.multi_train = False
            train_loader.dataset.transform = val_loader.dataset.transform
            self._network = self.trainer.model
            self._network.eval()
            embedding_list = []
            label_list = []
            with torch.no_grad():
                for i, batch in enumerate(train_loader):
                    data = batch['data'].cuda()
                    label = batch['label'].cuda()
    
                    b = data.size()[0]
                    m = data.size()[0] // b
                    labels = torch.stack([label*m+ii for ii in range(m)], 1).view(-1)
                    embedding = self._network.get_features(data)

                    embedding_list.append(embedding.cpu())
                    label_list.append(labels.cpu())
            embedding_list = torch.cat(embedding_list, dim=0)
            label_list = torch.cat(label_list, dim=0)

            proto_list = []

            for class_index in range(self.args.CIL.base_class*m):
                data_index = (label_list == class_index).nonzero()
                embedding_this = embedding_list[data_index.squeeze(-1)]
                embedding_this = embedding_this.mean(0)
                proto_list.append(embedding_this)

            proto_list = torch.stack(proto_list, dim=0)

            self._network.fc.weight.data[:self.args.CIL.base_class*m] = proto_list


    def test_step(self, trainer, data, label, *args, **kwargs):
        """Test step for standard supervised learning.

        Args:
            trainer (object): Trainer object.
            data (torch.Tensor): Input data.
            label (torch.Tensor): Label data.
            args (tuple): Additional arguments.
            kwargs (dict): Additional keyword arguments.

        Returns:
            results (dict): Test results. Contains the following:
                - "loss": Loss value.
                - "acc": Accuracy value.
        """
        return self.val_step(trainer, data, label, *args, **kwargs)

    def _incremental_train(self):
        return self._network

    def _accuracy(self, labels, preds):
        """
        compute accuracy 
        Args:
            labels: true label
            preds: predict label
        """
        correct = (preds == labels).sum().item()  # 统计预测正确的数量
        total = labels.size(0)  # 总样本数量
        acc = correct / total  # 计算 accuracy
        return acc