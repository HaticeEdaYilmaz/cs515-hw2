import argparse


def get_params():
    parser = argparse.ArgumentParser(description="Deep Learning on MNIST / CIFAR-10")

    parser.add_argument("--mode",      choices=["train", "test", "both"], default="both")
    parser.add_argument("--dataset",   choices=["mnist", "cifar10"],      default="cifar10")
    parser.add_argument("--model", choices=["mlp", "cnn", "vgg", "resnet", "mobilenet"], default="mlp")
    parser.add_argument("--epochs",    type=int,   default=10)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--device",    type=str,   default="cpu")
    parser.add_argument("--batch_size",type=int,   default=64)
    # VGG-specific
    parser.add_argument("--vgg_depth", choices=["11", "13", "16", "19"], default="16")
    # ResNet-specific: map a simple int to a block config
    parser.add_argument("--resnet_layers", type=int, nargs=4, default=[2, 2, 2, 2],
                        metavar=("L1", "L2", "L3", "L4"),
                        help="Number of blocks per ResNet layer (default: 2 2 2 2 = ResNet-18)")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--resize", type=int,default = 224)
    parser.add_argument("--option", type=int, choices=[1, 2], default=1)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--distill", action="store_true")
    parser.add_argument("--teacher_path", type=str, default="")

    args = parser.parse_args()

    if args.pretrained and args.option == 1:  # imagenet  
        input_size = 224   # 3 x 224 x 224
        mean = (0.485, 0.456, 0.406) 
        std = (0.229, 0.224, 0.225)
        
    else:                         # cifar10
        input_size = 3072         # 3 × 32 × 32
        mean       = (0.4914, 0.4822, 0.4465)
        std        = (0.2023, 0.1994, 0.2010)

    return {
        # Data
        "dataset":      args.dataset,
        "data_dir":     "./data",
        "num_workers":  2,
        "mean":         mean,
        "std":          std,

        # Model
        "model":        args.model,
        "input_size":   input_size,
        "hidden_sizes": [512, 256, 128],
        "num_classes":  10,
        "dropout":      0.3,
        "vgg_depth":    args.vgg_depth,
        "resnet_layers": args.resnet_layers,
        "pretrained": args.pretrained,
        "resize": args.resize,
        "option": args.option,

        # Training
        "epochs":        args.epochs,
        "batch_size":    args.batch_size,
        "learning_rate": args.lr,
        "weight_decay":  1e-4,
        "label_smoothing": args.label_smoothing,
        "distill": args.distill,
        "teacher_path": args.teacher_path,

        # Misc
        "seed":         42,
        "device":       args.device,
        "save_path":    "cnn_distilled.pth",
        "log_interval": 100,
        "load_path": "/content/cs515-hw2/best_model.pth",

        # CLI
        "mode":         args.mode,
    }