import os
import csv
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

# def normalizing(input_pc):
#     dat_ref = np.array(list(csv.reader(open("../data/reference.csv", 'r'))))  # dat_ref 따로
#     input_pc = np.vstack((np.array(input_pc[2:7]), np.array(input_pc[8:])))
#     temp_ind = []
#     for a in range(len(dat_ref[0])):
#         index_num = np.where(dat_ref[0, a] == input_pc[0, :])[0][0]
#         temp_ind.append(index_num)
#     sample = input_pc[:, temp_ind]
#     sample_vals = sample[5:].astype('float32')
#     sample_minmax = dat_ref[3:5].astype('float32')
#     sample_norm = (sample_vals - sample_minmax[0]) / (sample_minmax[1] - sample_minmax[0])
#     return sample_norm


def normalizing(input_pc):                  # Normalization
    dat_ref = np.array(list(csv.reader(open("../data/reference.csv", 'r'))))
    temp_ind = []
    for a in range(len(dat_ref[0])):
        index_num = np.where(dat_ref[0, a] == input_pc[0, :])[0][0]
        temp_ind.append(index_num)
    sample = input_pc[:, temp_ind]
    sample_vals = sample[1:].astype('float32')
    sample_minmax = dat_ref[3:5, :].astype('float32')
    sample_norm = (sample_vals - sample_minmax[0]) / (sample_minmax[1] - sample_minmax[0])
    return sample_norm


def generate_2d(sample, lag=5):                 # 3 Channel preprocessing
    surface_1 = []
    surface_2 = []
    for i in range(lag, sample.shape[0]):
        surface_1.append(sample[i])
        surface_2.append(sample[i - int(lag)] - sample[i])
    surface_1 = np.array(surface_1)
    surface_2 = np.array(surface_2)
    n_rows = surface_1.shape[0]
    n_cols = surface_1.shape[1]
    dat = np.concatenate((surface_1.reshape(n_rows, n_cols, 1), surface_2.reshape(n_rows, n_cols, 1)), axis=2)
    return dat


def generate_3d(sample, lag=5):                 # 3 Channel preprocessing
    surface_1 = []
    surface_2 = []
    surface_3 = []
    for i in range(3*lag, sample.shape[0]):
        surface_1.append(sample[i])
        # surface_1.append(sample[i - int(1 * lag)] - sample[i])
        surface_2.append(sample[i - int(2*lag)] - sample[i])
        surface_3.append(sample[i - int(3*lag)] - sample[i])
    surface_1 = np.array(surface_1)
    surface_2 = np.array(surface_2)
    surface_3 = np.array(surface_3)
    n_rows = surface_1.shape[0]
    n_cols = surface_1.shape[1]
    dat = np.concatenate((surface_1.reshape(n_rows, n_cols, 1), surface_2.reshape(n_rows, n_cols, 1), surface_3.reshape(n_rows, n_cols, 1)), axis=2)
    return dat


# def data_loader(file_name, folder='', nrows=60):  # data load
#     temp = open(os.path.join(folder, file_name), 'r', encoding='ISO-8859-1')
#     case_temp = list(csv.reader(temp))[:]
#     if '0000-00' in file_name:
#         print("normal")
#         temp_norm = normalizing(case_temp)[0:nrows]
#     else:
#         temp_norm = normalizing(case_temp)[11:11+nrows]
#     cls = file_name.split('-')[0] + '_' + file_name.split('-')[1]
#     cls_tile = np.tile(np.array([cls]), (temp_norm.shape[0], 1))
#     temp_norm = np.hstack((np.round(temp_norm, 6), cls_tile))
#     del temp, case_temp, cls, cls_tile  # memory
#     return temp_norm


def data_loader(file_name, folder='', start=600, step=180):          # data load
    df = pd.read_csv(folder+file_name, index_col=0, nrows=start+step)
    df = df.replace('.*', 0, regex=True).fillna(0)
    df = df.iloc[start:]

    case_ind = np.array(list(df))
    case_temp = df.to_numpy()
    case_temp = np.vstack([case_ind, case_temp])
    temp_norm = normalizing(case_temp)[:step]

    cls = file_name.split('_')[0] + '_' + file_name.split('_')[1]
    cls_tile = np.tile(np.array([cls]), (temp_norm.shape[0], 1))
    temp_norm = np.hstack((np.round(temp_norm, 6), cls_tile))
    del df, case_ind, case_temp, cls, cls_tile      # memory
    return temp_norm


def save_model(model, cfgs={}):
    if not os.path.exists("../models/" + cfgs['Data_src']):
        os.mkdir("../models/" + cfgs['Data_src'] + "/")
    model_to_save = model.to_json()
    model_dict = json.loads(model_to_save)
    model_dict[u'cfgs'] = cfgs
    model_to_save = json.dumps(model_dict)
    with open("../models/" + cfgs['Data_src'] + "/model[" + cfgs['configs_name'] + "].json", 'w') as json_file:
        json_file.write(model_to_save)
    # model.save_weights("../models/" + cfgs['Data_src'] + "/model[" + cfgs['configs_name'] + "].h5")
    print("model saved")


def load_model(model_fname):
    from tensorflow.keras.models import model_from_json
    f_model = open("../models/" + model_fname + "/model[" + model_fname + "].json", 'r')
    loaded_json = f_model.read()
    f_model.close()
    loaded_dict = json.loads(loaded_json)
    configs = loaded_dict['cfgs']
    loaded_model = model_from_json(loaded_json)
    loaded_model.load_weights("../models/" + model_fname + "/model[" + model_fname + "].h5")
    print("model loaded")
    return loaded_model, configs


def plot_acc(history, cfgs):
    fig_acc = plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.01)
    plt.legend(['Training', 'Validation'], loc='lower right')
    fig_acc.savefig('../results/' + cfgs['Data_src'] + '/Model_accuracy' + '.png')


def plt_loss(history, cfgs):
    fig_loss = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'], loc='upper right')
    fig_loss.savefig('../results/' + cfgs['Data_src'] + '/Model_loss' + '.png')


def plt_cm(true_labels, predicted_labels, cfgs):
    import itertools
    from sklearn.metrics import confusion_matrix as cm
    conf_mat = cm(true_labels, predicted_labels, range(len(cfgs['Abnorm_classes'])))
    print(conf_mat)

    fig_conf = plt.figure(figsize=(9, 8))
    plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(cfgs['Abnorm_classes']))
    plt.xticks(tick_marks, cfgs['Abnorm_classes'], rotation=45)
    plt.yticks(tick_marks, cfgs['Abnorm_classes'])

    threshold = conf_mat.max() / 2.
    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        plt.text(j, i, format(conf_mat[i, j], 'd'), fontsize=16, horizontalalignment='center',
                 color='white' if conf_mat[i, j] > threshold else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    fig_conf.tight_layout()
    fig_conf.savefig('../results/' + cfgs['Data_src'] + '/Confusion_matrix' + '.png')
    return conf_mat


def data_format(folder="khnp_temp", param_set="reference_all"): # Param_format 파일에 있는 변수만 저장. 포문당 하나씩, 메모리문제 해결.
    if not os.path.exists("../data/" + folder + "_transform"):
        os.mkdir("../data/" + folder + "_transform/")

    Param_format = np.array(list(csv.reader(open("../data/" + param_set + ".csv", 'r'))))
    for root, dirs, files in os.walk("../data/" + folder + "/"):
        files.sort()
        for i, fname in enumerate(files):
            print(i, fname)
            case_param = np.array(list(csv.reader(open("../data/" + folder + "/" + fname, 'r', encoding='ISO-8859-1')))[2:3])
            case = np.array(list(csv.reader(open("../data/" + folder + "/" + fname, 'r', encoding='ISO-8859-1')))[8:])
            pos_group = [0]
            for a in range(len(Param_format[0])):
                pos_temp = np.where(Param_format[0, a] == case_param[0, :])[0]
                if len(pos_temp) > 0 :
                    pos = pos_temp[0]
                    pos_group.append(pos)
                elif len(pos_temp) == 0 :
                    print("not exist", Param_format[0, a])
            case = case[600:780, pos_group]

            with open("../data/" + folder + "_transform/" + files[i], 'w', newline='') as f:
                wr = csv.writer(f)
                wr.writerows(case_param[:,pos_group])
                wr.writerows(case)
                f.close()


def param_format(): # xai에 정리된 parameter에 대해 ref 파일의 minmax 값으로 변경.
    xai = np.array(list(csv.reader(open("../data/select_param.csv", 'r'))))
    # xai = np.transpose(xai)
    print(xai.shape)

    ref = np.array(list(csv.reader(open("../data/reference.csv", 'r'))))
    print(ref.shape)

    ref_2 = []
    for i in range(xai.shape[1]):
        pos_temp = np.where(xai[0, i] == ref[0, :])[0]
        if len(pos_temp) > 0:
            pos = pos_temp[0]
            print(i, xai[0, i], ref[:, pos])
            ref_2.append(ref[:, pos])
    ref_2 = np.transpose(np.array(ref_2))

    with open("../data/reference_set.csv", 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerows(ref_2)
        f.close()


if __name__ == "__main__":
    print("Preprocessing")
    data_format("Example")
    # param_format()