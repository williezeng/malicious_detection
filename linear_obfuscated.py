import argparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
import csv_splitter
import torch
import yaml
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from models.logistic_regression import LogisticRegression
from models.simple_fcnn import simple_fcnn
import train

cwd = os.getcwd()
FULLY_CONNECTED_MODEL = 'fully_connected'
LOGISTIC_MODEL = 'logistic'
CHECKPOINT_NAME = 'best_model.pth'

def get_tensor(x_vals, labels):
    tensor_target = torch.tensor(labels.values, dtype=torch.float64)
    x_val_tensor = torch.tensor(x_vals.values, dtype=torch.float64)
    tensor_ds = torch.utils.data.TensorDataset(x_val_tensor, tensor_target)
    return tensor_ds


def train_data_reader(fname, args):
    read_df = pd.read_csv(os.path.join(args.input_dir, fname))
    print(fname + ' is loaded')
    yy = read_df["malicious"]  # only class id data
    read_df = read_df.drop("malicious", axis=1)  # everything other than class id data
    train_x, test_x, train_y, test_y = train_test_split(read_df, yy, test_size=0.15, random_state=3)
    del read_df
    del yy
    train_loader = torch.utils.data.DataLoader(dataset=get_tensor(train_x, train_y), batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=get_tensor(test_x, test_y), batch_size=args.batch_size, shuffle=True)
    return train_loader, test_loader


def test_data_reader(fname, batch_size):
    read_df = pd.read_csv(fname)
    print(fname + ' is loaded')
    yy = read_df["malicious"]  # only class id data
    read_df = read_df.drop("malicious", axis=1)  # everything other than class id data
    test_loader = torch.utils.data.DataLoader(dataset=get_tensor(read_df, yy), batch_size=batch_size, shuffle=True)
    return test_loader


def extract_existing_model(model, args):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    checkpoint = torch.load(os.path.join(cwd, 'best_models', args.model, CHECKPOINT_NAME))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    train_losses = checkpoint['t_loss']
    valid_losses = checkpoint['v_loss']
    return model, optimizer, train_losses, valid_losses


def perform_training(model, loss_fn, args, train_loader, test_loader, optimizer, train_losses, valid_losses):
    train_malicious_count = 0
    valid_malicious_count = 0

    # # Transfer the model to the GPU if available
    if torch.cuda.is_available():
        print("[*] Hardware: Cuda")
        model = model.cuda()
    else:
        print("[*] Hardware: CPU")

    for epoch in range(1, args.epochs + 1):
        best = 0
        train_loss = 0.0
        model.train()
        for xx_data, lbl in train_loader:
            xx_data = xx_data.cuda()
            lbl = lbl.cuda()
            optimizer.zero_grad()
            if epoch == 1:  # get the malicious count. You only need to perform this in one epoch (doesn't change)
                train_malicious_count += len(torch.nonzero(lbl))
            predict = model(xx_data.float())
            loss = loss_fn(predict, lbl.long())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xx_data.size(0)
        model.eval()
        acc, cm = train.validate(epoch, test_loader, model, loss_fn)
        if acc > best:
            best = acc
    print('[*] Best Prec @1 Acccuracy: {:.4f}'.format(best))

    train_malicious_percent = train_malicious_count/len(train_loader.sampler) * 100
    print('trained dataset has {} % malicious files'.format(train_malicious_percent))
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 't_loss': train_losses, 'v_loss': valid_losses},
               os.path.join(cwd, 'best_models', args.model, CHECKPOINT_NAME))


def test_model(dir_name, batch_size, model, loss_fn):
    best = 0.0
    best_cm = None
    for fname in os.listdir(dir_name):
        if fname.startswith("split_dataset_"):
            test_data = test_data_reader(os.path.join(dir_name, fname), batch_size)
            acc, cm = train.validate(1, test_data, model, loss_fn)
            if acc > best:
                best = acc
                best_cm = cm
    print('[*] Best Prec @1 Acccuracy: {:.4f}'.format(best))
    per_cls_acc = best_cm.diag().detach().numpy().tolist()
    for i, acc_i in enumerate(per_cls_acc):
        print("[*] Accuracy of Class {}: {:.4f}".format(i, acc_i))
    return best_cm

def training_iterator(model, args, loss_fn):
    test_acc = []
    for fname in os.listdir(args.input_dir):
        if fname.startswith("split_dataset_"):
            train_loader, test_loader = train_data_reader(fname, args)
            if os.path.exists(os.path.join(cwd, 'best_models', args.model, CHECKPOINT_NAME)):
                model, optimizer, train_losses, valid_losses = extract_existing_model(model, args)
            else:
                train_losses = []
                valid_losses = []
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
            perform_training(model, loss_fn, args, train_loader, test_loader, optimizer, train_losses, valid_losses)
            print("TESTING")
            loaded_model = torch.load(os.path.join(cwd, 'best_models', args.model, CHECKPOINT_NAME))
            model.load_state_dict(loaded_model['model_state_dict'])
            # Iterate and test on each dataset individually or you will be out of memory (OOM)
            test_acc.append(test_model(args.test_dir, args.batch_size, model, loss_fn))
    print('BEST TEST ACC {}'.format(max(test_acc))

def declare_classifier_model(learning_rate, momentum):
    opt = SGD(learning_rate=learning_rate, momentum=momentum)
    model = Sequential()
    model.add(Dense(10, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def declare_fully_connected_model(print=False):
    model = simple_fcnn()
    if print:
        print(model)
    return model

def declare_log_reg_model():
    D_in, D_out = 4998, 2
    return LogisticRegression(D_in, D_out)


def lr_best_guess(args):
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasClassifier
        # create model
    zsdf = KerasClassifier(build_fn=declare_classifier_model)
    # define the grid search parameters
    # TODO: ADD YOUR ADDITIONAL HYPERPARAMETERS HERE
    learning_rate = [0.0000000001]
    momentum = [0.80, 0.70]
    epochs = [50]
    param_grid = dict(epochs=epochs, learning_rate=learning_rate, momentum=momentum)
    grid = GridSearchCV(estimator=zsdf, param_grid=param_grid, cv=3)
    for fname in os.listdir(args.input_dir):
        if fname.startswith("split_dataset_"):
            read_df = pd.read_csv(os.path.join(args.input_dir, fname))
            print(fname + ' is loaded')
            yy = read_df["malicious"]  # only class id data
            read_df = read_df.drop("malicious", axis=1)  # everything other than class id data
            grid_result = grid.fit(read_df, yy)
            # summarize results
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            exit()


def parser():
    prsr = argparse.ArgumentParser()
    prsr.add_argument('--input_dir', type=str, dest='input_dir', default=csv_splitter.DEFAULT_DIR_NAME, help='The dir that contains the input csv file(s). The default is ' + csv_splitter.DEFAULT_DIR_NAME)
    prsr.add_argument('--test_dir', type=str, dest='test_dir', default=csv_splitter.DEFAULT_TEST_DIR, help='The dir that contains the test file(s). The default is ' + csv_splitter.DEFAULT_TEST_DIR)
    prsr.add_argument('--find_best', dest='find_best', default=False, action='store_true', help='toggle finding best parameters')
    prsr.add_argument('--model', dest='model', type=str, default=LOGISTIC_MODEL, choices=[FULLY_CONNECTED_MODEL, LOGISTIC_MODEL], help='Declare the model, default is {}'.format(LOGISTIC_MODEL))
    return prsr

def main():
    args = parser().parse_args()
    if args.model == FULLY_CONNECTED_MODEL:
        with open('configs/fully_connected.yaml') as f:
            config = yaml.load(f)
    else:
        with open('configs/logistic.yaml') as f:
            config = yaml.load(f)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)
    if not os.path.exists(args.input_dir):
        raise('The input path does not exist.')

    if args.loss_type == "CE":
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        raise Exception(f"[!] The {args.loss_type} loss function is not supported.")

    if args.find_best:
        lr_best_guess(args)
        exit()

    if args.model == FULLY_CONNECTED_MODEL:
        model = declare_fully_connected_model()
    if args.model == LOGISTIC_MODEL:
        model = declare_log_reg_model()

    # Iterate and train on each dataset individually or you will be OOM
    start_time = time.time()
    training_iterator(model, args, loss_fn)
    end_time = time.time()
    diff = (end_time - start_time)/60
    print("Time took {} minutes".format(diff))

if __name__ == '__main__':
    main()