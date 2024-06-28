import torch
import torch.nn as nn
import torch.optim as optim
from model import PneumoniaDetectionModel
from data_processing import get_data_loaders
from tqdm import tqdm
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader = get_data_loaders('data/train', 'data/val', 'data/test', batch_size=32)

model = PneumoniaDetectionModel(num_classes=2).to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

def train_epoch(model, criterion, optimizer, train_loader, device):
    model.train()
    train_loss = 0.0
    train_acc = 0

    for data, target in tqdm(train_loader, desc="Training", leave=False):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * data.size(0)
        _, pred = torch.max(output, dim=1)
        correct_tensor = pred.eq(target.data.view_as(pred))
        accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
        train_acc += accuracy.item() * data.size(0)

    train_loss /= len(train_loader.dataset)
    train_acc /= len(train_loader.dataset)

    return train_loss, train_acc

def validate_epoch(model, criterion, val_loader, device):
    model.eval()
    valid_loss = 0.0
    valid_acc = 0

    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Validation", leave=False):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item() * data.size(0)

            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            valid_acc += accuracy.item() * data.size(0)

    valid_loss /= len(val_loader.dataset)
    valid_acc /= len(val_loader.dataset)

    return valid_loss, valid_acc

if __name__ == "__main__":
    save_file_name = 'models/pneumonia_detection_model_best.pth'
    n_epochs = 10
    history = []
    model.epochs = 0

    for epoch in tqdm(range(n_epochs), desc="Training Epochs"):
        train_loss, train_acc = train_epoch(model, criterion, optimizer, train_loader, device)
        valid_loss, valid_acc = validate_epoch(model, criterion, val_loader, device)

        model.epochs += 1
        history.append([train_loss, valid_loss, train_acc, valid_acc])

        print(f'Epoch: {epoch+1} Train Loss: {train_loss:.4f} Val Loss: {valid_loss:.4f} Train Acc: {train_acc:.2f} Val Acc: {valid_acc:.2f}')

        torch.save(model.state_dict(), save_file_name)

    history = pd.DataFrame(history, columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Loss: {test_loss/len(test_loader):.4f}')
    print(f'Test Accuracy: {100 * correct / total:.2f}%')

    torch.save(model.state_dict(), 'models/pneumonia_detection_model.pth')
