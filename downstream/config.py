import argparse

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--message', '--msg', type=str, default=None)
    parser.add_argument('--train_path', type=str, default=None)
    parser.add_argument('--val_path', type=str, default=None)
    parser.add_argument('--test_path', type=str, default=None)

    parser.add_argument('--epochs', type=int, default=100, help="number of epochs")
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--w', type=int, default=16)
    
    parser.add_argument('--log_dir', type=str, default="runs")
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints")
    parser.add_argument('--print_freq', type=int, default=50)
    
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bits', type=int, default=8)
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--aug', type=bool, default=None)
    parser.add_argument('--weighted_CE', type=str, default='False')
    parser.add_argument('--csv_file', type=str, default=None)
    parser.add_argument('--downstream_name', type=str, default=None)
    parser.add_argument('--fine_tuning', type=bool, default=False)
    parser.add_argument('--bit', type=int, default=8)
    parser.add_argument('--num_class', type=int, default=6)
    parser.add_argument('--img_size', type=int, default=512)
    
    return parser.parse_args()
