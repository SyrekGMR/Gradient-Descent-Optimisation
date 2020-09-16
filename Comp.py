import torch
import os
import argparse
import torch.nn as nn
import torchvision
from torchvision import transforms
import time
import json
import Train
import Optimisers
import math

parser = argparse.ArgumentParser(description="PyTorch Training Code")

parser.add_argument("--num_epochs",
                    help="Number of epochs for training (Default=40)",
                    type=int,
                    default=40)
parser.add_argument("--cuda",
                    help="Use CUDA? (Default=True)",
                    type=bool,
                    default=True)
parser.add_argument("--dataset",
                    help="Name of dataset to be used (Default=CIFAR100)",
                    default="CIFAR100",
                    type=str
                    )
parser.add_argument("--dataset_path",
                    default=os.path.join(".", "Data"),
                    help="Path to custom dataset if requires (Default=None)")
parser.add_argument("-l", "--lr",
                    type= lambda s: [float(p) for p in s.split(",")],
                    default=None,
                    help="List of LR to iterate over during training (Default=None)")
parser.add_argument("--train",
                    default=True,
                    type=bool)
parser.add_argument("--test",
                    default=False,
                    type=bool)
parser.add_argument("--batch_size",
                    type=int,
                    default=64,
                    help="Batch size to use for training (Default=64)")
parser.add_argument("--optimizers",
                    type= lambda s: [str(p) for p in s.split(",")],
                    default=None,
                    help="List of Optimizers to iterate over during training (Default=None)")
parser.add_argument("--optimizer",
                    default="SGD",
                    type=str,
                    help="Str() -- Optimizer to use for training (Default=SGD)")
parser.add_argument("--model",
                    default="resnet18",
                    help="Model to be optimized, lowercase all (Default=resnet18)")
parser.add_argument("--model_save_path",
                    default=None,
                    type=str,
                    help="Path to save model")
parser.add_argument("--results_save_path",
                    default=os.path.join(".", "Results"),
                    help="Path to save results")
parser.add_argument("--model_load_path",
                    default=None,
                    help="Str() Path to load model checkpoint (Default=None)",)
parser.add_argument("--last_layer", 
                    default=None,
                    help="Str() name of last layer in model if using CIFAR")

args = parser.parse_args()


device = torch.device('cuda') if args.cuda else torch.device('cpu')

# Prepare datasets using torchvision with standard data augmentation techniques

if args.train:
    data = Data.data_prep(args.dataset, train=True)
    loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=args.batch_size)

if args.test:
    data = Data.data_prep(args.dataset, train=False)
    loader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=1)

print(f"Data Loaded: \n{data}")


# Parse optimiser inputs for list of optimisers to iterate over

try:
    optimizers = [getattr(torch.optim, opt) if opt in dir(torch.optim) else getattr(Optimizers, opt) for opt in args.optimizers]
except AttributeError:
    raise AttributeError(f"Invalid Optimizer Call! Optimizer not in torch.optim or Optimizers directory \n The optimizers called were {args.optimizers} \n The optimizers allowed are {[p for p in dir(torch.optim) if '__' not in p] + [p for p in dir(Optimizers) if '__' not in p]}")

print(f"Loaded Optimisers: {optimizers}")

n_epochs = args.num_epochs

criterion = torch.nn.CrossEntropyLoss()

main_track = {}

for optimizer in optimizers:
    main_track = {}
    
    for lr in args.lr:

        # Modify last layer of model if using CIFAR dataset

        model = getattr(torchvision.models, args.model)(pretrained=False)
        if "CIFAR" in args.dataset:
            out = 10 if args.dataset=="CIFAR10" else 100
            getattr(model, args.last_layer) = nn.Linear(getattr(model, args.last_layer).in_features, out).to(device)
            model.fc = nn.Linear(model.fc.in_features, 100).to(device)


        if args.model_load_path:
            assert os.file.exists(args.model_load_path), f"Invalid checkpoint path => {args.model_load_path}"
            model.load_state_dict(torch.load(args.model_load_path))
        else:
            pass

        if "Lookahead" in str(optimizer):
            base_opt = torch.optim.Adam(model.parameters(), lr=lr)
            opt_config = {"optimizer": base_opt}
        elif "SVRG" in str(optimizer):
            opt_config = {"params": model.parameters(), "nbatches": math.ceil(len(train_data)/args.batch_size), "lr":lr}
        else:
            opt_config = {"params": model.parameters(), "lr": lr}

        opt = optimizer(**opt_config)

        main_track[str(opt).split(" ")[0]] = []
        
        loss_track = []
        cosine_track = []
        
        for epoch in range(34, n_epochs + 1):
            
            # Perform one epoch of training

            loss, cosine =  Train.train(model, 
                                        opt, 
                                        main_track, 
                                        lr, 
                                        train_loader,
                                        device, 
                                        criterion, 
                                        epoch, 
                                        n_epochs= n_epochs)
            
            loss_track.append(loss)
            cosine_track.append(cosine)

            if args.model_save_path:
                torch.save(model.state_dict(), os.path.join(args.model_save_path, (str(epoch) + "_" + str(lr) + ".pth")))

        if args.test:
            acc = evaluate(model, device, test_loader)       
            main_track[str(opt).split(" ")[0]] = [loss_track, cosine_track, acc]  
        else:
            main_track[str(opt).split(" ")[0]] = [loss_track, cosine_track]  

        
        name = str(opt).split(" ")[0] + str(model.__class__.__name__) + "_" + str(lr) + ".json"
        path = os.join(args.results_save_path, name)
        with open(path, "w") as f:
            json.dump(main_track, f)
