import sys
sys.path.append("../Python37")
import keras
from keras.utils import np_utils
from keras.models import Sequential, model_from_json
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Flatten, Dense, Activation, Dropout, ReLU, LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from preprocessing import *


def Training(src="example", ep=10) :
    print("Training")

    cfgs = dict()
    cfgs['n_epochs'] = ep
    cfgs['Data_src'] = src
    cfgs['configs_name'] = cfgs['Data_src']
    cfgs['Abnorm_classes'] = []

    train_folder = '../data/' + cfgs['Data_src'] + '/'
    train_list = os.listdir(train_folder)

    dat = []
    for num, train_file in enumerate(train_list):
        temp = data_loader(train_file, train_folder, 0, 120)
        dat.append(temp)
    dat = np.array(dat)
    print("data loaded")

    # classes = np.array(dat[:, :, -1].reshape(-1, 120, 1))
    classes = np.array(dat[:, -1, -1]).reshape(-1, 1)   #
    abnorm_classes = np.unique(dat[:,-1,-1])

    cfgs['Abnorm_classes'] = abnorm_classes.tolist()
    cfgs['n_classes'] = abnorm_classes.shape[0]

    dat = dat[:, :, :-1].astype('float32')
    ys=[]
    Y = classes.astype('str')
    for yy in Y:
        ys.append(np.where(abnorm_classes == yy[0])[0][0])
    Y = np.array(ys)
    Y = np_utils.to_categorical(Y, cfgs['n_classes'])

    print(dat.shape, Y.shape)
    dat_train, dat_val, Y_train, Y_val = train_test_split(dat, Y, test_size=0.2, random_state=1, stratify=Y)

    # dat_train
    X_train = generate_2d(dat_train[0], lag=5)
    for d in dat_train[1:]:
        X_train = np.vstack((X_train, generate_2d(d, 5)))
    Y_train = np.repeat(Y_train, 115, axis=0)
    # dat_val
    X_val = generate_2d(dat_val[0], lag=5)
    for d in dat_val[1:]:
        X_val = np.vstack((X_val, generate_2d(d, 5)))
    Y_val = np.repeat(Y_val, 115, axis=0)
    print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape)

    # X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=1)
    input_shape = X_train[0].shape

    print("-----------------------------------------------------------------")
    print("Model:2-CH 1-D CNN", " Number of epochs:", cfgs['n_epochs'], " Label:", cfgs['Abnorm_classes'])
    print("-----------------------------------------------------------------")

    if not os.path.exists("../models/" + cfgs['Data_src']):
        os.mkdir("../models/" + cfgs['Data_src'] + "/")

    checkpoint = ModelCheckpoint("../models/" + cfgs['Data_src'] + "/model[" + cfgs['configs_name'] + "].h5", monitor='val_loss', verbose=2,
    save_best_only=True, mode='auto', period=1)

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20)
    # leaky_relu = tf.nn.leaky_relu
    model = Sequential()
    # model.add(Conv1D(16, kernel_size=3, input_shape=input_shape))
    # model.add(LeakyReLU(alpha=0.05))
    # model.add(Conv1D(32, kernel_size=3))
    # model.add(LeakyReLU(alpha=0.05))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Conv1D(64, kernel_size=3))
    # model.add(LeakyReLU(alpha=0.05))
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))    
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(units=cfgs['n_classes'], activation='softmax'))
    # model.add(Activation('softmax'))
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(model.summary())
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=cfgs['n_epochs'], batch_size=64,
                        verbose=2, callbacks=[checkpoint, early_stopping])
    save_model(model, cfgs=cfgs)

    if not os.path.exists("../results/" + cfgs['Data_src']):
        os.mkdir("../results/" + cfgs['Data_src'] + "/")

    prediction = model.predict(X_val)
    predicted_labels = np.argmax(prediction, axis=1)

    plot_acc(history=history, cfgs=cfgs)
    plt_loss(history=history, cfgs=cfgs)
    confusion_matrix = plt_cm(np.argmax(Y_val, axis=1), predicted_labels, cfgs)

    confusion_matrix = np.vstack((cfgs['Abnorm_classes'], confusion_matrix))
    with open("../results/" + cfgs['Data_src'] + "/Confusion_matrix.csv", 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerows(confusion_matrix)
        f.close()

    return True


if __name__ == "__main__":
    Training(src="Example_transform", ep=30)