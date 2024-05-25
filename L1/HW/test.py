import csv
from torch.utils.data import DataLoader

from config import *
from util import *
from dataset import COVID19Dataset
from model import COVIDModel
import data


def save_pred(preds, file):
    """ Save predictions to specified file """
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])


train_data, valid_data, test_data = data.read_data()
x_train, x_valid, x_test, y_train, y_valid = data.select_feat(train_data, valid_data, test_data)

test_dataset = COVID19Dataset(x_test)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

model = COVIDModel(input_size=x_train.shape[1]).to(device)
model.load_state_dict(torch.load(config['save_path']))
preds = predict(test_loader, model, device)
save_pred(preds, 'predict.csv')
