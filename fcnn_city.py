import numpy as np
import pickle
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
from my_adam_optimizer import MyAdamOptimizer


class ToTensor(nn.Module):
    def __init__(self):
        super().__init__()
        self.map_to_train_classes = {x.id: x.train_id for x in datasets.Cityscapes.classes}

    def __call__(self, target):
        '''
        Convert to tensor, add undefined class.
        '''
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        mask = torch.zeros_like(target)
        for id, train_id in self.map_to_train_classes.items():
            if train_id == 255:
                mask[target == id] = 19
            else:
                mask[target == id] = train_id
        return mask


def main():
    num_epochs = 20

    # Load and normalize data
    transform = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.2869, 0.3251, 0.2839),
                                                         (0.1761, 0.1810, 0.1777))])
    target_transform = transforms.Compose([transforms.Resize((256, 256)),
                                           transforms.CenterCrop(224),
                                           ToTensor()])
    train_data = datasets.Cityscapes('./data/cityscapes', split='train',
                                     mode='coarse',
                                     target_type='semantic',
                                     transform=transform,
                                     target_transform=target_transform)

    print(len(train_data))
    data_loader = torch.utils.data.DataLoader(train_data,
                                              batch_size=3,
                                              shuffle=True,
                                              num_workers=4)

    results = {'Adam': [], 'SGDNesterov': [], 'AdaGrad': []}
    colors = {'Adam': 'r', 'SGDNesterov': 'g', 'AdaGrad': 'b'}
    for name in ['Adam', 'SGDNesterov', 'AdaGrad']:
        print('Starting', name)

        model = models.segmentation.fcn_resnet50(pretrained=False,
                                                 num_classes=20)
        model.cuda()

        criterion = nn.CrossEntropyLoss().cuda()

        if name == 'Adam':
            optimizer = MyAdamOptimizer(model.parameters(), weight_decay=0.01, lr=0.0001)
        elif name == 'SGDNesterov':
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9,
                                  weight_decay=0.01, nesterov=True)
        elif name == 'AdaGrad':
            optimizer = optim.Adagrad(model.parameters(), weight_decay=0.01, lr=0.001)

        for i in tqdm(range(num_epochs)):
            loss = 0
            for x, y in tqdm(data_loader):
                x = x.cuda().type(torch.float32)
                y = y.cuda()
                optimizer.zero_grad()
                y_pred = model(x)['out']
                train_loss = criterion(y_pred, y)
                train_loss.backward()
                optimizer.step()
                loss += train_loss.item()
            print(loss/len(data_loader))
            results[name].append(loss/len(data_loader))
        #results[name] = pickle.load(open('results/cityscapes/' + name + '.p', 'rb'))
        torch.save(model, name+'.pt')
        plt.plot(np.arange(num_epochs), results[name], label=name, c=colors[name], linewidth=1)
        pickle.dump(results[name], open(name+'.p', 'wb'))

    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.xticks(np.arange(num_epochs))
    plt.title('Cityscapes FCNN')
    plt.legend()
    plt.tight_layout()
    plt.savefig('out.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
