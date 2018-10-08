import keras
import sys
sys.path.append("../..")
from tcn import tcn

from utils import data_generator


x_train, y_train = data_generator(601, 10, 30000)
x_test, y_test = data_generator(601, 10, 6000)

class PrintSomeValues(keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs={}):
        print(f'x_test[0:1] = {x_test[0:1].flatten()}.')
        print(f'y_test[0:1] = {y_test[0:1].flatten()}.')
        print(f'p.shape = {self.model.predict(x_test[0:1]).shape}.')
        print(f'p(x_test[0:1]) = {self.model.predict(x_test[0:1]).argmax(axis=2).flatten()}.')


def run_task(args):
    

    print(sum(x_train[0].tolist(), []))
    print(sum(y_train[0].tolist(), []))

    model = tcn.compiled_tcn(num_feat=1,
                             num_classes=10,
                             nb_filters=10,
                             kernel_size=8,
                             dilations=[2 ** i for i in range(9)],
                             nb_stacks=2,
                             max_len=x_train[0:1].shape[1],
                             activation='norm_relu',
                             use_skip_connections=True,
                             return_sequences=True)

    print(f'x_train.shape = {x_train.shape}')
    print(f'y_train.shape = {y_train.shape}')

    psv = PrintSomeValues()

    # Using sparse softmax.
    # http://chappers.github.io/web%20micro%20log/2017/01/26/quick-models-in-keras/
    model.summary()

    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=args.epochs,
              callbacks=[psv], batch_size=args.batch_size)
    
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

if __name__ == '__main__':

    args = prepare_task()
    
    run_task(args)
