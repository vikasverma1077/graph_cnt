import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='GcnInformax Arguments.')
    parser.add_argument('--DS', dest='DS', help='Dataset')
    parser.add_argument('--local', dest='local', action='store_const', 
            const=True, default=False)
    parser.add_argument('--glob', dest='glob', action='store_const', 
            const=True, default=False)
    parser.add_argument('--prior', dest='prior', action='store_const', 
            const=True, default=False)
    
    parser.add_argument('--epochs', dest='epochs', type=int, default=20,
            help='')
    parser.add_argument('--lr', dest='lr', type=float, default=0.01,
            help='Learning rate.')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=4,
            help='Number of graph convolution layers before each pooling')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=32,
            help='')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=128,
            help='')
    parser.add_argument('--temperature', dest='temperature', type=float, default=1.0,
            help='')
    parser.add_argument('--mixup_alpha', dest='mixup_alpha', type=float, default=0.9,
            help='')

    return parser.parse_args()

