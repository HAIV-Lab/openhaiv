# import new Network name here and add in model_class args
from .Network import MYNET
from utils import *
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import svm

from matplotlib.font_manager import FontProperties

from collections import defaultdict

font_path = ''

# plt.rcParams["font.sans-serif"] = "SimHei" # 用来正常显示中文标签SimHei
# plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

from losses import SupContrastive

def draw_confusion_matrix(label_true, label_pred, label_name, title="Confusion Matrix", pdf_save_path="./confusion_matrix", dpi=300):
    """
    @param label_true: 真实标签，比如[0,1,2,7,4,5,...]
    @param label_pred: 预测标签，比如[0,5,4,2,1,4,...]
    @param label_name: 标签名字，比如['cat','dog','flower',...]
    @param title: 图标题
    @param pdf_save_path: 是否保存，是则为保存路径pdf_save_path=xxx.png | xxx.pdf | ...等其他plt.savefig支持的保存格式
    @param dpi: 保存到文件的分辨率，论文一般要求至少300dpi
    @return:
    example：
            draw_confusion_matrix(label_true=y_gt,
                          label_pred=y_pred,
                          label_name=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
                          title="Confusion Matrix on Fer2013",
                          pdf_save_path="Confusion_Matrix_on_Fer2013.png",
                          dpi=300)

    """
    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize='true')

    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name, rotation=45)

    plt.tight_layout()

    plt.colorbar()
    
    font_size = 8
    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
            value = float(format('%.2f' % cm[j, i]))
            if value >0.02:
                plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color, fontsize=font_size)

    # plt.show()
    if not pdf_save_path is None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)
        plt.clf()


def base_train(model, trainloader, criterion, optimizer, scheduler, epoch, transform, args):
    tl = Averager()
    tl_joint = Averager()
    # tl_moco = Averager()
    # tl_moco_global = Averager()
    # tl_moco_small = Averager()
    ta = Averager()
    model = model.train()
    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        data_path, data, single_labels = [_ for _ in batch]
        b, c, h, w = data[1].shape
        original = data[0].cuda(non_blocking=True)
        data[1] = data[1].cuda(non_blocking=True)
        data[2] = data[2].cuda(non_blocking=True)
        single_labels = single_labels.cuda(non_blocking=True)
        #args.num_crop=[2,4]
        if len(args.num_crops) > 1:
            data_small = data[args.num_crops[0]+1].unsqueeze(1)
            for j in range(1, args.num_crops[1]):
                data_small = torch.cat((data_small, data[j+args.num_crops[0]+1].unsqueeze(1)), dim=1)
            #变成了128*4，就是直接batch_size变成了512
            data_small = data_small.view(-1, c, args.size_crops[1], args.size_crops[1]).cuda(non_blocking=True)
        else:
            data_small = None
        
        data_classify = transform(original)    
        data_query = transform(data[1])
        data_key = transform(data[2])
        data_small = transform(data_small)
        m = data_query.size()[0] // b
        joint_labels = torch.stack([single_labels*m+ii for ii in range(m)], 1).view(-1)
        
        joint_preds,_= model(im_cla=data_classify)
        # im_q=data_query, im_k=data_key, labels=joint_labels, im_q_small=data_small
       #output_global, output_small, target_global, target_small
       # loss_moco_global = criterion(output_global, target_global)
        #loss_moco_small = criterion(output_small, target_small)
        #loss_moco = args.alpha * loss_moco_global + args.beta * loss_moco_small

        joint_preds = joint_preds[:, :args.base_class*m]
        joint_loss = F.cross_entropy(joint_preds, joint_labels)

        agg_preds = 0
        for i in range(m):
            agg_preds = agg_preds + joint_preds[i::m, i::m] / m

        loss = joint_loss#+0*loss_moco
        total_loss = loss
        
        acc = count_acc(agg_preds, single_labels)

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        tl_joint.add(joint_loss.item())
        #tl_moco_global.add(loss_moco_global.item())
       # tl_moco_small.add(loss_moco_small.item())
       # tl_moco.add(loss_moco.item())
        ta.add(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    tl_joint = tl_joint.item()
    #tl_moco = tl_moco.item()
    #tl_moco_global = tl_moco_global.item()
    #tl_moco_small = tl_moco_small.item()
    return tl, tl_joint, ta ##, tl_moco, tl_moco_global, tl_moco_small,


def replace_base_fc(trainset, test_transform, data_transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = test_transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data_path = batch[0]
            data= batch[1].cuda()
            label = batch[2].cuda()
            # data, label = [_.cuda() for _ in batch]
            b = data.size()[0]
            data = data_transform(data)
            m = data.size()[0] // b
            labels = torch.stack([label*m+ii for ii in range(m)], 1).view(-1)
            model.mode = 'encoder'
            embedding = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(labels.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class*m):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.fc.weight.data[:args.base_class*m] = proto_list

    return model

def compute_sample_dis(trainloader, data_transform, model, args):

    class_features = {}
    data_pathes = defaultdict(list)
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data_path= batch[0]
            data = batch[1].cuda()
            labels = batch[2]

            data = data_transform(data)
            features, _ = model.encode_q(data)

            for index, j in enumerate(labels):
                j = j.item()
                if j not in class_features.keys():
                    class_features[j]=features[index].reshape(1, -1)
                    data_pathes[j].append(data_path[index])
                else:
                    class_features[j]=torch.cat((class_features[j], features[index].reshape(1, -1)), dim=0)
                    data_pathes[j].append(data_path[index])
    protos = {}
    for key in class_features.keys():
        protos[key] = torch.mean(class_features[key], dim=0)
    
    distances = defaultdict(list)
    for key in class_features.keys():
        for i, feature in enumerate(class_features[key]):
            dis = torch.norm(class_features[key][i] - protos[key], p=2).detach().cpu().numpy()
            distances[key].append((data_pathes[key][i], dis))
    sorted_distances = defaultdict(list)
    
    for key in distances.keys():
        sorted_list = sorted(distances[key], key=lambda x: x[1])
        sorted_distances[key]=sorted_list[:5]
    print("*************************************")
    
    print(sorted_distances)

    print("*************************************")


def update_fc_ft(trainloader, data_transform, model, m, session, args):
    # incremental finetuning
    old_class = args.base_class + args.way * (session - 1)
    new_class = args.base_class + args.way * session 
    new_fc = nn.Parameter(
        torch.rand(args.way*m, model.num_features, device="cuda"),
        requires_grad=True)
    new_fc.data.copy_(model.fc.weight[old_class*m : new_class*m, :].data)
    
    if args.dataset == 'mini_imagenet':
        optimizer = torch.optim.SGD([{'params': new_fc, 'lr': args.lr_new},
                                     {'params': model.encoder_q.fc.parameters(), 'lr': 0.05*args.lr_new},
                                     {'params': model.encoder_q.layer4.parameters(), 'lr': 0.001*args.lr_new},],
                                    momentum=0.9, dampening=0.9, weight_decay=0)
        
    if args.dataset == 'cub200':
        optimizer = torch.optim.SGD([{'params': new_fc, 'lr': args.lr_new}],
                                    momentum=0.9, dampening=0.9, weight_decay=0)
    if args.dataset == 'remote':
        # optimizer = torch.optim.SGD([{'params': new_fc, 'lr': args.lr_new}],
        #                             momentum=0.9, dampening=0.9, weight_decay=0)
        optimizer = torch.optim.SGD([{'params': new_fc, 'lr': args.lr_new},
                                     {'params': model.encoder_q.fc.parameters(), 'lr': 0.05 * args.lr_new},
                                     {'params': model.encoder_q.layer4.parameters(), 'lr': 0.002 * args.lr_new}, ],
                                    momentum=0.9, dampening=0.9, weight_decay=0)

    elif args.dataset == 'cifar100':
        optimizer = torch.optim.Adam([{'params': new_fc, 'lr': args.lr_new},
                                      {'params': model.encoder_q.fc.parameters(), 'lr': 0.01*args.lr_new},
                                      {'params': model.encoder_q.layer3.parameters(), 'lr':0.02*args.lr_new}],
                                      weight_decay=0)
        
    criterion = SupContrastive().cuda() 

    with torch.enable_grad():
        for epoch in range(args.epochs_new):
            for batch in trainloader:
                data, single_labels = [_ for _ in batch]
                b, c, h, w = data[1].shape
                origin = data[0].cuda(non_blocking=True)
                data[1] = data[1].cuda(non_blocking=True)
                data[2] = data[2].cuda(non_blocking=True)
                single_labels = single_labels.cuda(non_blocking=True)
                if len(args.num_crops) > 1:
                    data_small = data[args.num_crops[0]+1].unsqueeze(1)
                    for j in range(1, args.num_crops[1]):
                        data_small = torch.cat((data_small, data[j+args.num_crops[0]+1].unsqueeze(1)), dim=1)
                    data_small = data_small.view(-1, c, args.size_crops[1], args.size_crops[1]).cuda(non_blocking=True)
                else:
                    data_small = None
            data_classify = data_transform(origin)    
            data_query = data_transform(data[1])
            data_key = data_transform(data[2])
            data_small = data_transform(data_small)
            joint_labels = torch.stack([single_labels*m+ii for ii in range(m)], 1).view(-1)
            
            old_fc = model.fc.weight[:old_class*m, :].clone().detach()    
            fc = torch.cat([old_fc, new_fc], dim=0)
            features, _ = model.encode_q(data_classify)
            features.detach()
            logits = model.get_logits(features,fc)
            joint_loss = F.cross_entropy(logits, joint_labels)
            _, output_global, output_small, target_global, target_small = model(im_cla=data_classify, im_q=data_query, im_k=data_key, labels=joint_labels, im_q_small=data_small, base_sess=False, last_epochs_new=(epoch==args.epochs_new-1))
            loss_moco_global = criterion(output_global, target_global)
            loss_moco_small = criterion(output_small, target_small)
            loss_moco = args.alpha * loss_moco_global + args.beta * loss_moco_small 
            loss = joint_loss + loss_moco         
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.fc.weight.data[old_class*m : new_class*m, :].copy_(new_fc.data)


def test(model, testloader, epoch, transform, args, session):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        preds = []
        labels=[]
        features=[]
        images_paths = []
        for i, batch in enumerate(tqdm_gen, 1):
            data_path = batch[0]
            data= batch[1].cuda()
            test_label = batch[2].cuda()
            # data_path, data, test_label = [_.cuda() for _ in batch]
            b = data.size()[0]
            data = transform(data)
            m = data.size()[0] // b
            joint_preds, feature = model(data)
            joint_preds = joint_preds[:, :test_class*m]
            
            agg_preds = 0
            agg_feature = 0
            for j in range(m):
                agg_preds = agg_preds + joint_preds[j::m, j::m] / m
                agg_feature = agg_feature+ feature[j::m, j::m] / m
            predicts = torch.max(agg_preds, dim=1)[1]
            preds.append(predicts)
            labels.append(test_label)
            features.append(agg_feature)
            for j in range(len(data_path)):
                images_paths.append(data_path[j])
            # images_paths.append(data_path)
            loss = F.cross_entropy(agg_preds, test_label)
            acc = count_acc(agg_preds, test_label)

            vl.add(loss.item())
            va.add(acc)
        preds = torch.cat(preds,dim=0)
        labels= torch.cat(labels,dim=0)
        features = torch.cat(features, dim=0)
        X =  features.cpu()
        y=labels.cpu()
        # 创建并拟合 t-SNE 模型，将特征降维到2维

        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)
    
        # 获取不同类别的索引
        unique_labels = np.unique(y)

        # 根据类别绘制不同颜色的散点图
        for label in unique_labels:
            indices = np.where(y == label)
            plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], label=f'Class {label}')

        plt.xlabel('')
        plt.ylabel('')
        plt.xticks([])
        plt.yticks([])
        plt.title('')
        plt.legend(loc='upper left', prop={'size': 5})
        plt.show()
        plt.savefig(f"./{session}_test_class_{test_class}_epoch_{epoch}.jpg")
        plt.clf()
        if session == 0:
            draw_confusion_matrix(label_true=labels.cpu(), label_pred=preds.cpu(),
                                  label_name=['1', '2', '3', '4', '5', '6', '7', '8','9','10', '11'],
                                  title="混淆矩阵",
                                  pdf_save_path="Confusion_Matrix_Base.jpg",
                                  dpi=300)
        if session == 1:
            draw_confusion_matrix(label_true=labels.cpu(), label_pred=preds.cpu(),
                                  label_name=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'],
                                  title="混淆矩阵",
                                  pdf_save_path="Confusion_Matrix.jpg",
                                  dpi=300)
            correct_class_image_path = {}
            for _cls in torch.unique(labels):
                if _cls.item() >=8:
                    # 只看增量类别
                    correct_class_image_path[_cls.item()] = []
                    cls_indices = torch.where(labels==_cls)
                    cls_image_paths = []
                    # print("##############################################")
                    # print("cls_indices:", cls_indices)
                    # print("images_paths", images_paths)
                    # print("##############################################")
                    for i in range(len(cls_indices[0])):
                        cls_image_paths.append(images_paths[cls_indices[0][i].item()])
                    cls_predictions = preds[cls_indices]
                    cls_labels = labels[cls_indices]
                    cls_rlt = cls_predictions == cls_labels
                    for i in range(len(cls_rlt)):
                        if cls_rlt[i] == True:
                            correct_class_image_path[_cls.item()].append(cls_image_paths[i])
            
            # 打印正确识别的图片
            for _cls in correct_class_image_path.keys():
                print(f"Class {_cls} correct images: {correct_class_image_path[_cls]}")

        # 获取所有类别
        classes = torch.unique(labels)
        #, '35', '36', '37', '38', '39', '40'
        # 计算每个类别的准确率
        accuracy_per_class = {}
        for cls in classes:
            # 获取属于当前类别的预测结果和真实标签
            cls_indices = torch.where(labels == cls)
            cls_predictions =  preds[cls_indices]
            cls_labels = labels[cls_indices]

            # 计算当前类别的准确率
            correct = torch.sum(cls_predictions == cls_labels).item()
            total = len(cls_labels)
            accuracy = correct / total

            # 存储当前类别的准确率
            accuracy_per_class[cls.item()] = accuracy

        # 打印每个类别的准确率
        for cls, accuracy in accuracy_per_class.items():
            print(f"Class {cls}: Accuracy = {accuracy}")
        

        vl = vl.item()
        va = va.item()
    print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

    return vl,va




# 假设你有一个特征矩阵 X，其中每行表示一个样本，每列表示一个特征