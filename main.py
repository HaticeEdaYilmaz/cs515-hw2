import random
import ssl
import numpy as np
import torch

from parameters import get_params
from models.MLP import MLP
from models.CNN import MNIST_CNN, SimpleCNN
from models.VGG import VGG
from models.ResNet import ResNet, BasicBlock
from models.mobilenet import MobileNetV2
from train import run_training
from test  import run_test
from torchvision import models
import torch.nn as nn

# Fix for macOS SSL certificate verification error when downloading MNIST
ssl._create_default_https_context = ssl._create_unverified_context


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(params):
    model_name = params["model"]
    dataset    = params["dataset"]
    nc         = params["num_classes"]
    pretrained = params["pretrained"]
    

  
    if pretrained:
        if model_name == "vgg":
            if dataset == "mnist":
                raise ValueError("VGG pretrained expects 3-channel images. Use CIFAR-10.")
            
            model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            if params["option"] == 1:
                num_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(num_features,nc)
                for name,param in model.named_parameters():
                    if "classifier.6" not in name:
                        param.requires_grad = False

            elif params["option"] == 2:
              
                remove_idx = [16, 23, 30] # remove last 3 maxpool layers
                model.features = nn.Sequential(*[
                    layer for i, layer in enumerate(model.features)
                    if i not in remove_idx
                ])

                model.avgpool = nn.Identity() # remove avgpool

                model.classifier[0] = nn.Linear(512 * 8 * 8, 4096)
                model.classifier[6] = nn.Linear(4096, nc)

                for param in model.parameters():
                    param.requires_grad = True
                    
            
        else:
            raise ValueError(f"Pretrained {model_name} not supported. Use vgg.")

        return model


    if model_name == "mlp":
        return MLP(
            input_size   = params["input_size"],
            hidden_sizes = params["hidden_sizes"],
            num_classes  = nc,
            dropout      = params["dropout"],
        )

    if model_name == "cnn":
        if dataset == "mnist":
            return MNIST_CNN(num_classes=nc)
        else:
            return SimpleCNN(num_classes=nc)

    if model_name == "vgg":
        if dataset == "mnist":
            raise ValueError("VGG is designed for 3-channel images; use CIFAR-10 with VGG.")
        return VGG(dept=params["vgg_depth"], num_class=nc)

    if model_name == "resnet":
        if dataset == "mnist":
            raise ValueError("ResNet is designed for 3-channel images; use CIFAR-10 with ResNet.")
        return ResNet(BasicBlock, params["resnet_layers"], num_classes=nc)

    if model_name == "mobilenet":
        if dataset == "mnist":
            raise ValueError("MobileNetV2 is designed for 3-channel images; use CIFAR-10 with MobileNet.")
        return MobileNetV2(num_classes=nc)

    raise ValueError(f"Unknown model: {model_name}")




def main():
    params = get_params()

    set_seed(params["seed"])
    print(f"Seed set to: {params['seed']}")
    print(f"Dataset: {params['dataset']}  |  Model: {params['model']}")

    device = torch.device(
        params["device"] if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {device}")

    model = build_model(params).to(device)
    print(model)

    if params["mode"] in ("train", "both"):
        run_training(model, params, device)

    if params["mode"] in ("test", "both"):
        run_test(model, params, device)


if __name__ == "__main__":
    main()