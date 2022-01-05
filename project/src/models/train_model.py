# Run using
# python src/models/train_model.py -h --lr 0.001

import argparse
import sys
import torch
from model import MyAwesomeModel

def train():
    print("Training day and night")
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--lr', default=0.001)
    # add any additional argument that you want
    args = parser.parse_args(sys.argv[2:])
    print(args)
    
    model = MyAwesomeModel()
    images_train = torch.load('data\\processed\\images_train.pt')
    labels_train = torch.load('data\\processed\\labels_train.pt')
    train_set = torch.utils.data.TensorDataset(images_train, labels_train)
    #test = torch.utils.data.TensorDataset(images_test, labels_test)
    criterion = torch.nn.CrossEntropyLoss()
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    loss_lst = []
    for e in range(5):  
        running_loss = 0
        for images, labels in trainloader:
            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            print("Training loss: ", running_loss)
            loss_lst.append(running_loss)
    
    torch.save(model.state_dict(), 'models\\checkpoint.pth')

        
if __name__ == '__main__':
    train()
    
    
    
    
    
    
    
    
    