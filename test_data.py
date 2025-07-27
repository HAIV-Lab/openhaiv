import os
from pprint import pprint
from ncdia_old.utils import *
from ncdia.dataloader import get_cil_dataloader, get_transform, DataManager


"""
python test_data.py --config \ 
configs/increment/savc_remote.yml \ 
configs/dataloader/savc_att_remote.yml \ 
configs/networks/resnet18_savc_att.yml \ 
configs/pipelines/train_savc_att_base.yml \ 
--seed 1
"""
if __name__ == "__main__":
    cfg = setup_config()
    cil_path = "/new_data/cyf/CIL_Dataset"
    crop_transform, secondary_transform = get_transform(cfg.dataloader)
    datamanager = DataManager(
        "remote",
        cil_path,
        shuffle=False,
        seed=0,
        init_cls=10,
        increment=2,
        use_path=True,
        attr_path=os.path.join(cil_path, "Attribute0421.xlsx"),
    )
    # datamanager = DataManager("remote", cil_path, shuffle=False, seed=0, init_cls=10, increment=2, use_path=True,
    #                           crop_transform=crop_transform, secondary_transform=secondary_transform, attr_path=os.path.join(cil_path, "Attribute0421.xlsx"))

    pprint(datamanager._get_trsf("train"))
    from torchvision.transforms import ToPILImage, Compose

    for session in range(3):
        tset, trainloader, testloader = get_cil_dataloader(
            cfg, datamanager, session=session
        )
        data_sample = tset[0]
        pprint(data_sample["attribute"])
        print(data_sample["imgpath"], data_sample["label"], data_sample["data"].shape)

        # for idx in range(len(data_sample['data'])):
        #     pic = ToPILImage()(data_sample['data'][idx])
        #     pic.save("sample_{}.jpg".format(idx))
