import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from model import darknet

def main():
    imageNet_label = [line.strip() for line in open("demo/imagenet.shortnames.list", 'r')]

    dataset = dset.ImageFolder(root="demo/samples/",
                           transform=transforms.Compose([
                               transforms.Resize((448, 448)),
                               transforms.ToTensor()
                           ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    darknet19 = darknet.Darknet19(pretrained=True)
    darknet19.eval()

    for data, _ in dataloader:
        output = darknet19.forward(data)
        answer = int(torch.argmax(output))
        print("Class: {}({})".format(imageNet_label[answer],answer))
        plt.imshow(np.array(np.transpose(data[0], (1, 2, 0))))
        plt.show()

if __name__ == "__main__":
    main()
