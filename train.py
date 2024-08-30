import torch
from tqdm import tqdm 

def batch_train(model, device, train_loader, val_loader, optimizer, criterion, scheduler): 
    model.train()
    size = 0 
    correct = 0  
    running_loss = 0.0 
    with tqdm(total=len(train_loader), desc="Training", unit="batch") as pbar:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            correct += (output.argmax(1) == target).type(torch.float).sum().item()
            running_loss += loss.item()
            size += len(data) 

            pbar.set_postfix({'loss': running_loss / (batch_idx + 1), 'accuracy': 100 * correct / size})
            pbar.update(1)

    acc = (100 * correct) / size
    avg_loss = running_loss / (batch_idx + 1)
    print(f"Train: Avg loss: {avg_loss:.6f}, Accuracy: {acc:.2f}%")


    model.eval()
    running_loss = 0.0
    correct = 0
    size = 0

    with tqdm(total=len(val_loader), desc="Validation", unit="batch") as pbar:
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)

                output = model(data)
                loss = criterion(output, target)
                running_loss += loss.item()
                correct += (output.argmax(1) == target).type(torch.float).sum().item()
                size += len(data) 
                
                pbar.set_postfix({'val_loss': running_loss / (batch_idx + 1), 'val_accuracy': 100 * correct / size})
                pbar.update(1)

    vacc = (100 * correct) / size
    avg_vloss = running_loss / (batch_idx + 1)
    print(f"Validation: Avg loss: {avg_vloss:.6f}, Accuracy: {vacc:.2f}%")
    
    scheduler.step()

    return avg_loss, avg_vloss, acc, vacc