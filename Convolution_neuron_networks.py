import torch
from torch import nn,optim
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, transforms

transform=transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5,),(0.5,))])

train_data_MNIST=datasets.MNIST(root="./data",
                                train=True,
                                transform=transform,
                                download=False)
test_data_MNIST=datasets.MNIST(root="./data",
                               train=False,
                               transform=transform,
                               download=False)

train_loader_MNIST=DataLoader(dataset=train_data_MNIST,
                              batch_size=32,
                              shuffle=True)
test_loader_MNIST=DataLoader(dataset=test_data_MNIST,
                             batch_size=32,
                             shuffle=True)

class Convolutional_neuron_networks(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,stride=1,padding=1),# out_size=1,64,28,28
                                nn.BatchNorm2d(64),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),# out_size=1,64,28,28
                                nn.BatchNorm2d(64),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2,stride=2,padding=0),# out_size=1,64,14,14

                                nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),# out_size=1,128,14,14
                                nn.BatchNorm2d(128),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1),# out_size=1,128,14,14
                                nn.BatchNorm2d(128),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2,stride=2,padding=0),# out_size=1,128,7,7

                                nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),# out_size=1,256,7,7
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),# out_size=1,256,7,7
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),# out_size=1,256,7,7
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2,stride=2,padding=0),# out_size=1,256,3,3

                                nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1),# out_size=1,512,3,3
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),# out_size=1,512,3,3
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),# out_size=1,512,3,3
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2,stride=2)# out_size=1,512,1,1
                                )
        self.flaten=nn.Flatten()# out_size=512x1x1
        self.fc=nn.Sequential(nn.Linear(in_features=512,out_features=1024),
                              nn.ReLU(),nn.Dropout(0.5),
                              nn.Linear(in_features=1024,out_features=2048),
                              nn.ReLU(),nn.Dropout(0.5),
                              nn.Linear(in_features=2048,out_features=10))

    def forward(self,x):
        x=self.conv(x)
        x=self.flaten(x)
        x=self.fc(x)
        return x


if __name__=="__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=Convolutional_neuron_networks().to(device=device)
    num_epochs=2
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(params=model.parameters(),lr=0.002,betas=(0.9,0.999))

    for epoch in range(num_epochs):
        model.train()
        loss_total=0
        for image,label in train_loader_MNIST:
            image,label=image.to(device),label.to(device)
            optimizer.zero_grad()
            output=model(image)
            loss=criterion(output,label)
            loss.backward()
            optimizer.step()
            loss_total+=loss

        print(f"Epoch: {epoch+1}; Loss: {loss_total/len(train_loader_MNIST):.4f}")
    torch.save(model.state_dict(),"Convolutional_neuron_networks_WEIGHT.pth")
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for image,label in test_loader_MNIST:
            image,label=image.to(device),label.to(device)
            outputs=model(image)
            _,predicted=torch.max(outputs,1)
            total+=label.size(0)
            correct+=(predicted==label).sum().item()
    accuracy=100*correct/total
    print("độ chính xác trên bộ test: ",accuracy)