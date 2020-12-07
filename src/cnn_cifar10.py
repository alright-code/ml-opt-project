import numpy as np
import pickle
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
from my_adam_optimizer import MyAdamOptimizer


def main():
    num_epochs = 30

    # Load and normalize data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                         (0.2023, 0.1994, 0.2010))])
    train_data = datasets.CIFAR10('data/CIFAR/', download=True,
                                  train=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(train_data,
                                              batch_size=128,
                                              shuffle=True,
                                              num_workers=4)

    results = {'Adam': [], 'SGDNesterov': [], 'AdaGrad': []}
    colors = {'Adam': 'r', 'SGDNesterov': 'g', 'AdaGrad': 'b'}
    for name in ['Adam', 'SGDNesterov', 'AdaGrad']:
        print('Starting', name)

        # CNN from paper
        model = nn.Sequential(nn.Dropout(.1),
                              nn.Conv2d(3, 64, 5, padding=2), nn.ReLU(True),
                              nn.MaxPool2d(3, stride=2),
                              nn.Conv2d(64, 64, 5, padding=2), nn.ReLU(True),
                              nn.MaxPool2d(3, stride=2),
                              nn.Conv2d(64, 128, 5, padding=2), nn.ReLU(True),
                              nn.MaxPool2d(3, stride=2),
                              nn.Flatten(),
                              nn.Linear(1152, 1000), nn.ReLU(True),
                              nn.Dropout(.1),
                              nn.Linear(1000, 10))
        model.cuda()

        criterion = nn.CrossEntropyLoss()

        if name == 'Adam':
            optimizer = MyAdamOptimizer(model.parameters(), weight_decay=0, lr=0.0005)
        elif name == 'SGDNesterov':
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                                  weight_decay=0, nesterov=True)
        elif name == 'AdaGrad':
            optimizer = optim.Adagrad(model.parameters(), weight_decay=0, lr=0.0015)

        # Train
        for i in tqdm(range(num_epochs)):
            loss = 0
            for x, y in data_loader:
                x = x.cuda()
                y = y.cuda()
                optimizer.zero_grad()
                y_pred = model(x)
                train_loss = criterion(y_pred, y)
                train_loss.backward()
                optimizer.step()
                loss += train_loss.item()
            results[name].append(loss/len(data_loader))

        print(results[name][-1])
        plt.plot(np.arange(num_epochs), results[name], label=name, c=colors[name], linewidth=1)
        pickle.dump(results[name], open(name+'.p', 'wb'))

    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.xticks(np.arange(num_epochs))
    plt.title('CIFAR-10 CNN')
    plt.xticks(np.arange(0, num_epochs + 1, 5))
    plt.legend()
    plt.savefig('out.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
