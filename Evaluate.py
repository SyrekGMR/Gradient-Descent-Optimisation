import torch

def evaluate(model, device, test_data_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_data_loader:
            images = images.to(device)
            labels = labels.to(device)

            output = model.forward(images)
            output = torch.max(output, 1)[-1]
            correct += torch.sum(output == labels).item()

            total += len(output)

    return correct/total
