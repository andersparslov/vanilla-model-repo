# Run using 
# python src/models/predict_model.py -h --load_model_from models\\checkpoint.pth --eval_dir data\\processed\\
import torch
import argparse
import sys
from model import MyAwesomeModel


def evaluate():
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--load_model_from', default="models\\checkpoint.pth")
        parser.add_argument('--eval_dir', default="data\\processed\\")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        model = MyAwesomeModel()
        model.load_state_dict(torch.load(args.load_model_from))
        images_train = torch.load(args.eval_dir + 'images_test.pt')
        labels_train = torch.load(args.eval_dir + 'labels_test.pt')
        test_set = torch.utils.data.TensorDataset(images_train, labels_train)
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
    evaluate()
    