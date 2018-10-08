from utils import data_generator
import sys
sys.path.append("../..")
from tcn import tcn

def prepare_task():
    import os
    import argparse
    parser = argparse.ArgumentParser(description="TCN network on copy task.")
    parser.add_argument('-g', '--gpu_id', default="0", help="GPU device form 0-7")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    
    args = parser.parse_args()
    print(args)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print("made save dir:" + args.save_dir)
        
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    print("use gpu id:" + args.gpu_id)
    
    return args

def run_task(args):
    (x_train, y_train), (x_test, y_test) = data_generator()

    model = tcn.compiled_tcn(return_sequences=False,
                             num_feat=1,
                             num_classes=10,
                             nb_filters=25,
                             kernel_size=7,
                             dilations=[2 ** i for i in range(9)],
                             nb_stacks=2,
                             max_len=x_train[0:1].shape[1],
                             activation='norm_relu',
                             use_skip_connections=True)

    print(f'x_train.shape = {x_train.shape}')
    print(f'y_train.shape = {y_train.shape}')
    print(f'x_test.shape = {x_test.shape}')
    print(f'y_test.shape = {y_test.shape}')

    model.summary()

    model.fit(x_train, y_train.squeeze().argmax(axis=1), epochs=args.epochs,
              validation_data=(x_test, y_test.squeeze().argmax(axis=1)))


if __name__ == '__main__':
    args = prepare_task()
    run_task(args)
