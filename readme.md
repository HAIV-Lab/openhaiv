# 一、如何运行

希望利用 scripts 中的 sh 直接运行。



# 二、代码结构
现在各个文件夹中，几乎都只有savc-att相关的文件是编写完成的，其他的文件多是占位
1. config：存放 config 文件，暂时有三种，dataloader 指定遥感的数据的相关 args，pipieline 指定训练过程中的参数，network 指定模型的参数
2. 训练new阶段时记得修改对应的pipeline config里面的pretrain文件路径。
3. main.py 主文件，利用 pipeline.run() 作用主接口
4. ncdia（novel class discovery with attribute）
   - augmentation 存放数据增强方式域代码
   - dataloader 数据读取，返回 trainset, trainloader, testloader
   - discovers 准备存放所有的新类别发现，没写
   - evaluator 存放所有的评估代码，比如 ood 测试，属性、分类测试
   - loss 存放损失函数
   - network
   - pipelines 是 main 函数运行的主要入口，包括整个代码流程，没写完
   - recorders 是结果运行的记录者，print
   - trainers 是每一个 epoch 的训练代码