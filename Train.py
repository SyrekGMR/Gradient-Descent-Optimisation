import json
import torch
from Evaluate import evaluate
import time

def cosine(t1, t2):
    cos = torch.nn.CosineSimilarity(dim=0)
    c = cos(t1.detach().cpu().flatten(), t2.detach().cpu().flatten())
    
    return c



def train(model, optimizer, main_track, lr, data_loader, 
          device, criterion, epoch, cosine=False, batch_size=128, n_epochs=20):

    iterations_left = (n_epochs - epoch) * len(data_loader)
    total_iterations = n_epochs * len(data_loader) 

    start_time = time.time()

    cosine_track = []

    for i, (images, labels) in enumerate(data_loader):

        
        if i > 1 and cosine:
            prior_grad = [p.grad.data.detach().cpu() for p in model.parameters()]
        
        s = time.time()

        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()

        if optimizer.__class__.__name__ == "SVRG": 
            optimizer.step(i)

        else:
            optimizer.step()


        cosine_ = "NA"

        if i > 1 and cosine:
            cos = torch.nn.CosineSimilarity(dim=0)
            c_track = []
            for g1, p1 in zip(prior_grad, model.parameters()):
                c_track.append(cosine(g1, p1.grad).item())            

            cosine_ = round(sum(c_track) / len(c_track), 2)
            cosine_track.append(cosine_)

        e = time.time()
        taken = round((e - s), 2)

        remaining = (taken) * iterations_left
        iterations_left -= 1
        
        print(f"""OPTIMIZER: {str(optimizer).split(' ')[0]}| LR: {lr}| BATCH_SIZE: {data_loader.batch_size}| EPOCH: {epoch}/{n_epochs}| ITERATION: {i}/{len(data_loader)}| LOSS: {round(loss.item(), 3)}| COSINE: {cosine_}| REMAINING: {round(remaining, 2)} second(s)\r""", end="")

    # Returns loss at the end of epoch and cosine track if used

    return loss.item(), cosine_track


