import argparse

from trainer.train_all import train_all

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    train_all(args.config)
