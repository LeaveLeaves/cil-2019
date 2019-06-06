# from eval import *
from dataload import get_train_loader

def check_config():
    from config import config

    for k, v in config.items():
        print("{:20}: {}".format(k, v))

def check_eval():
    print("Checking eval")

if __name__ == "__main__":
    check_config()