import argparse
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def parse_args():
    arg_parser = argparse.ArgumentParser(description='unsupervised motion', add_help=False)
    arg_parser.add_argument('--train', action='store_true')
    arg_parser.add_argument('--test', action='store_true')
    arg_parser.add_argument('--method', default='unsupervised')
    arg_parser.add_argument('--train_epoch', type=int, default=1000)
    arg_parser.add_argument('--test_epoch', type=int, default=10)
    arg_parser.add_argument('--test_interval', type=int, default=500)
    arg_parser.add_argument('--display', action='store_true')
    arg_parser.add_argument('--save_dir', default='./model')
    arg_parser.add_argument('--batch_size', type=int, default=32)
    arg_parser.add_argument('--init_model_path', default='')
    arg_parser.add_argument('--learning_rate', type=float, default=0.01)
    arg_parser.add_argument('--motion_range', type=int, default=1)
    arg_parser.add_argument('--image_size', type=int, default=11)
    args = arg_parser.parse_args()
    return args
