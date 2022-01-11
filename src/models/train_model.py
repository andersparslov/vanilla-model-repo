# Run using
# python src/models/train_model.py --lr 0.001

import argparse
import sys
import torch
import wandb
import helper
from model import MyAwesomeModel


EVAL_DIR = "data\\processed\\"

def train():
    print("Using GPU: ", torch.cuda.is_available())
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--optimizer', default="adam")
    # add any additional argument that you want
    args = parser.parse_args()
    print(args)
    wandb.init(config=args)
    
    model = MyAwesomeModel()
    wandb.watch(model, log_freq=100)
    images_train = torch.load('data/processed/images_train.pt')
    labels_train = torch.load('data/processed/labels_train.pt')
    train_set = torch.utils.data.TensorDataset(images_train, labels_train)
    #test = torch.utils.data.TensorDataset(images_test, labels_test)
    criterion = torch.nn.CrossEntropyLoss()
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=float(args.lr))
    loss_lst = []
    for e in range(int(args.epochs)):  
        running_loss = 0
        for images, labels in trainloader:
            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            # Validation 
            images_test = torch.load(EVAL_DIR + 'images_test.pt')
            labels_test = torch.load(EVAL_DIR + 'labels_test.pt')
            test_set = torch.utils.data.TensorDataset(images_test, labels_test)
            testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
            
            correct_count = 0
            for images, labels in testloader:
                with torch.no_grad():
                    y = torch.argmax(model(images), dim=1)
                    correct_count += torch.sum(y == labels)
            accuracy = correct_count / len(test_set)
            wandb.log({"validation_accuracy" : accuracy,
                       "loss" : running_loss})
            print("Training loss: ", running_loss, f'Validation accuracy: {accuracy.item()*100}%')
            loss_lst.append(running_loss)
    
    torch.save(model.state_dict(), 'models/checkpoint.pt')
    # Predict example (for logging image to wandb)
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    for i in range(10):
        img = images[i].unsqueeze(0)
        # Turn off gradients to speed up this part
        with torch.no_grad():
            logps = model(img)

        # Output of the network are log-probabilities, need to take exponential for probabilities
        ps = torch.exp(logps)
        fig = wandb.Image(helper.view_classify(img.view(1, 28, 28), ps))
        wandb.log({"prediction" : fig})

    
if __name__ == '__main__':
    train()
    
    
    
    
    
    
    
    
    