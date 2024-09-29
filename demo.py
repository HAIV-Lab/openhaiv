from ncdia.dataloader.datasets.remoteiv import Remoteiv
from ncdia.dataloader import MergedDataset

if __name__ == '__main__':
    traindataset = Remoteiv("/data/gjq/Dataset/VAIS/Fine_paired/vis", "/data/gjq/Dataset/VAIS/Fine_paired/ir", "train", transform="train")
    hist_testset = MergedDataset()
    hist_testset.merge([traindataset], True)
    print(hist_testset.images, hist_testset.labels, hist_testset.transform)