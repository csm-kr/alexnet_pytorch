import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epoch', type=int, default=90)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--vis_step', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--gpu_ids', nargs="+", default=['0'])
    # usage : --gpu_ids 0, 1, 2, 3
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Alexnet training', parents=[get_args_parser()])
    args = parser.parse_args()
    print(args)
