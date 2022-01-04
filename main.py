import argparse
import sys
import torch
from data import mnist
from model import MyAwesomeModel


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        model = MyAwesomeModel()
        train_set, _ = mnist()
        criterion = torch.nn.CrossEntropyLoss()
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
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
        torch.save(model.state_dict(), 'checkpoint.pth')
        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="checkpoint.pth")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        model = MyAwesomeModel()
        model.load_state_dict(torch.load(args.load_model_from))
        _, test_set = mnist()
        testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
        
         # Validation acc
        correct_count = 0
        for images, labels in testloader:
            with torch.no_grad():
                y = torch.argmax(model(images), dim=1)
                correct_count += torch.sum(y == labels)
        accuracy = correct_count / len(test_set)
        print(f'Accuracy: {accuracy.item()*100}%')
        
if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    