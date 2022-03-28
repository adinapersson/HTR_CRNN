from calendar import c
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

np.random.seed(42)
tf.random.set_seed(42)
base_path = "../data"
dataset_folder = "full_data_org" # Choose which dataset to use
base_image_path = os.path.join(base_path,dataset_folder)


def loadData(): # Returns information of samples split in train val and test sets
    lines_words = []
    words_path = os.path.join(base_path,"words.txt")
    words = open(words_path, "r",encoding="utf-8").readlines()
    # Exclude error segmentaions from dataset
    for line in words:
        if line[0] == "#":
            continue
        if line.split(" ")[1] != "err":
            lines_words.append(line.strip())

    np.random.shuffle(lines_words)

    # Split data in training validation and test sets
    train_size = 0.9
    split_idx = int(train_size * len(lines_words))
    train_data = lines_words[:split_idx]
    test_data = lines_words[split_idx:]

    val_split_idx = int(0.5 * len(test_data))
    validation_data = test_data[:val_split_idx]
    test_data = test_data[val_split_idx:]

    assert len(lines_words) == len(train_data) + len(validation_data) + len(
        test_data
    )
    return train_data, test_data, validation_data


def get_image_paths_and_labels(data): 
    paths = []
    labels = []
    for file_line in enumerate(data):
        line_split = file_line.split(" ")

        # Each line split will have this format for the corresponding image:
        # part1/part1-part2/part1-part2-part3.png
        image_name = line_split[0]
        partI = image_name.split("-")[0]
        partII = image_name.split("-")[1]
        img_path = os.path.join(
            base_image_path, partI, partI + "-" + partII, image_name + ".png"
        )
        if os.path.getsize(img_path):
            paths.append(img_path)
            labels.append(line_split[8])

    return paths, labels


def get_max_length_and_chars(labels): # Find maximum length and the size of the vocabulary in the training data.
    characters = set()
    max_length = 0

    for label in labels:
        for char in label:
            characters.add(char)

        max_length = max(max_length, len(label))
    print("Maximum length: ", max_length)
    print("Vocab size: ", len(characters))
    return max_length, sorted(characters)


def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 1)
    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def vectorize_label(label):
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    length = tf.shape(label)[0]
    pad_amount = max_length - length
    label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=padding_token)
    return label


def process_images_labels(image_path, label):
    image = preprocess_image(image_path)
    label = vectorize_label(label)
    return {"image": image, "label": label}


def prepare_dataset(image_paths, labels, batch_size):
    AUTOTUNE = tf.data.AUTOTUNE
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(
        process_images_labels, num_parallel_calls=AUTOTUNE
    )
    return dataset.batch(batch_size).cache().prefetch(AUTOTUNE)


def viewSamples(train_ds):
    for data in train_ds.take(1):
        images, labels = data["image"], data["label"]

        _, ax = plt.subplots(4, 4, figsize=(15, 8))

        for i in range(16):
            img = images[i]
            img = tf.image.flip_left_right(img)
            img = tf.transpose(img, perm=[1, 0, 2])
            img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
            img = img[:, :, 0]

            # Gather indices where label!= padding_token.
            label = labels[i]
            indices = tf.gather(label, tf.where(tf.math.not_equal(label, padding_token)))
            # Convert to string.
            label = tf.strings.reduce_join(num_to_char(indices))
            label = label.numpy().decode("utf-8")

            ax[i // 4, i % 4].imshow(img, cmap="gray")
            ax[i // 4, i % 4].set_title(label)
            ax[i // 4, i % 4].axis("off")

    plt.show()


class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions.
        return y_pred


def build_model(image_width, image_height, dropout):
    # Inputs to the model Keras
    input_img = keras.Input(shape=(image_width, image_height, 1), name="image")
    labels = layers.Input(name="label", shape=(None,))

    # Harald
    cnn = 8
    if cnn==5:
        kernel_vals = [5,5,3,3,3]
        feature_vals = [32,64,128,128,256]
        pool_vals = [(2,2),(2,2),(1,2),(1,2),(1,2)]
        conv_names = ["Conv1","Conv2","Conv3","Conv4","Conv5",]
        pool_names = ["Pool1","Pool2","Pool3","Pool4","Pool5",]
        dropout_vals = [0, 0, 0, 0, 0]
    if cnn == 8:
        kernel_vals = [5,5,5,5,3,3,3,3]
        feature_vals = [32,32,64,64,128,128,128,256]
        pool_vals = [(1,1),(2,2),(1,1),(2,2),(1,2),(1,1),(1,2),(1,2)]
        conv_names = ["Conv1","Conv2","Conv3","Conv4","Conv5","Conv6","Conv7","Conv8",]
        pool_names = ["Pool1","Pool2","Pool3","Pool4","Pool5","Pool6","Pool7","Pool8",]
        # dropout_vals = [0, 0, 0, 0, 0.1, 0.15, 0.2, 0.2]
        dropout_vals = [0, 0, 0, 0, 0, 0, 0, 0]
    num_convs = len(kernel_vals)

    x = input_img
    for i in range(num_convs):
        x = layers.Conv2D(
        feature_vals[i],
        kernel_vals[i],
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name=conv_names[i],
        )(x)
        x = layers.MaxPooling2D(pool_vals[i], name=pool_names[i])(x)
        x = layers.Dropout(dropout_vals[i])(x)
    
    x = layers.Reshape(target_shape=(32,256), name="reshape1")(x)
    # x = layers.Dropout(0.33)(x)

    # RNNs
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=dropout))(x)
    # x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.25))(x)

    x = layers.Reshape(target_shape=(1,32,512), name="reshape2")(x)

    x = layers.Conv2D(len(char_to_num.get_vocabulary()) + 1,1, activation="softmax", name="conv_last")(x)
    x = layers.Reshape(target_shape=(32,len(char_to_num.get_vocabulary()) + 1), name="reshape3")(x) # Fix names for get layers? 

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model
    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
    )
    # Optimizer
    opt = keras.optimizers.Adam()
    # Compile the model and return
    model.compile(optimizer=opt)
    return model


def calculate_edit_distance(labels, predictions):
    # Get a single batch and convert its labels to sparse tensors.
    saprse_labels = tf.cast(tf.sparse.from_dense(labels), dtype=tf.int64)

    # Make predictions and convert them to sparse tensors.
    input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
    predictions_decoded = keras.backend.ctc_decode(
        predictions, input_length=input_len, greedy=True
    )[0][0][:, :max_length]
    sparse_predictions = tf.cast(
        tf.sparse.from_dense(predictions_decoded), dtype=tf.int64
    )

    # Compute individual edit distances and average them out.
    edit_distances = tf.edit_distance(
        sparse_predictions, saprse_labels, normalize=False
    )
    return tf.reduce_mean(edit_distances)


class EditDistanceCallback(keras.callbacks.Callback):
    def __init__(self, pred_model, validation_images, validation_labels):
        super().__init__()
        self.prediction_model = pred_model
        self.validation_images = validation_images
        self.validation_labels = validation_labels

    def on_epoch_end(self, epoch, logs=None):
        edit_distances = []

        for i in range(len(self.validation_images)):
            labels = self.validation_labels[i]
            predictions = self.prediction_model.predict(self.validation_images[i])
            edit_distances.append(calculate_edit_distance(labels, predictions).numpy())

        print(
            f" Mean edit distance for epoch {epoch + 1}: {np.mean(edit_distances):.4f}"
        )


def train_model(model,train_ds,validation_ds,epochs,model_name,toTrain,toLoad):
    
    validation_images = []
    validation_labels = []

    for batch in validation_ds:
        validation_images.append(batch["image"])
        validation_labels.append(batch["label"])

    # Get the prediction model by extracting layers till the output layer
    prediction_model = keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="reshape3").output 
    )


    # Early stopping
    early_stopping_patience = 10
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
    )


    edit_distance_callback = EditDistanceCallback(prediction_model, validation_images, validation_labels)

    model_folder = "../saved_models"
    # Load weights from previous run
    if toLoad:
        model = keras.models.load_model(os.path.join(model_folder, model_name))
    
    # Train the model.
    if toTrain:
        history = model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=epochs,
            callbacks=[edit_distance_callback,early_stopping],
        )
        try:
            os.mkdir(model_folder)
        except OSError as error:
            print('Folder already exists')
        model.save(os.path.join(model_folder,model_name))


    else:
        prediction_model = keras.models.Model(
            model.get_layer(name="image").input, model.get_layer(name="reshape3").output
        )

    return prediction_model


def accuracy(validation_ds,prediction_model):
    accuracy = 0
    for batch in validation_ds:
        batch_labels = batch["label"]
        batch_images = batch["image"]

        preds = prediction_model.predict(batch_images)
        pred_texts = decode_batch_predictions(preds)

        gt_texts = []
        for label in batch_labels:
            indices = tf.gather(label, tf.where(tf.math.not_equal(label, padding_token)))
            # Convert to string
            label = tf.strings.reduce_join(num_to_char(indices))
            gt_texts.append(label.numpy().decode("utf-8"))
        
        output = []
        for i in range(len(pred_texts)):
            output.append(pred_texts[i] == gt_texts[i])
        accuracy += sum(output)/len(gt_texts)
    return 100*accuracy/len(validation_ds)


def decode_batch_predictions(pred): # A utility function to decode the output of the network.
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]
    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


def viewResults(test_ds, prediction_model):
    #  Let's check results on some test samples.
    for batch in test_ds.take(3):
        batch_images = batch["image"]
        batch_labels = batch["label"]
        _, ax = plt.subplots(4, 4, figsize=(15, 8))

        preds = prediction_model.predict(batch_images)
        pred_texts = decode_batch_predictions(preds)

        gt_texts = []
        for label in batch_labels:
            indices = tf.gather(label, tf.where(tf.math.not_equal(label, padding_token)))
            # Convert to string
            label = tf.strings.reduce_join(num_to_char(indices))
            gt_texts.append(label.numpy().decode("utf-8"))

        for i in range(16):
            img = batch_images[i]
            img = tf.image.flip_left_right(img)
            img = tf.transpose(img, perm=[1, 0, 2])
            img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
            img = img[:, :, 0]

            title = f"Prediction: {pred_texts[i]} \n gt_text: {gt_texts[i]}"
            ax[i // 4, i % 4].imshow(img, cmap="gray")
            ax[i // 4, i % 4].set_title(title)
            ax[i // 4, i % 4].axis("off")

    plt.show()


def writeReport(acc,epochs,dropout,model_name,num_train):
    report_folder = "../reports"
    try:
        os.mkdir(report_folder)
    except OSError as error:
        print('Folder already exists')

    now = datetime.now()
    reportname = now.strftime("report-%m-%d-%H-%M-%S.txt")
    f = open(os.path.join(report_folder,reportname), "w")
    f.writelines(f"Epochs={epochs} \nAccuracy={acc}% \nDropout={dropout} \nNumber of training samples={num_train} \nModel name={model_name}")



def main():
    global max_length, char_to_num, num_to_char, padding_token

    batch_size = 64
    epochs = 90
    dropout = 0.50
    padding_token = 99
    image_width = 128
    image_height = 32

    model_name = "model_cnn8_dropout0" # "model2_aug_org_75_0.5_8"
    toTrain = 1
    toLoad = 0

    train_data, test_data, validation_data = loadData()


    print(f"Total training samples: {len(train_data)}")
    print(f"Total test samples: {len(test_data)}")
    print(f"Total validation samples: {len(validation_data)}")


    train_img_paths, train_labels = get_image_paths_and_labels(train_data)
    validation_img_paths, validation_labels = get_image_paths_and_labels(validation_data)
    test_img_paths, test_labels = get_image_paths_and_labels(test_data)


    max_length, characters = get_max_length_and_chars(train_labels)

    print(characters)
    print("number of characters",len(characters))

    # Mapping characters to integers.
    char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)

    # Mapping integers back to original characters.
    num_to_char = StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )

    train_ds = prepare_dataset(train_img_paths, train_labels, batch_size)
    validation_ds = prepare_dataset(validation_img_paths, validation_labels, batch_size)
    test_ds = prepare_dataset(test_img_paths, test_labels, batch_size)

    # View some samples from training dataset
    #viewSamples(train_ds)

    # Get model
    model = build_model(image_width, image_height, dropout)
    model.summary()

    # Train model
    prediction_model = train_model(model,train_ds,validation_ds,epochs,model_name,toTrain,toLoad)


    # Compute accuracy
    acc = accuracy(validation_ds,prediction_model)
    print(f"validation accuracy = {acc}" )
    acc = accuracy(test_ds,prediction_model)
    print(f"test accuracy = {acc}" )

    # Write a report of the settings used and the result
    writeReport(acc,epochs,dropout,model_name,len(train_data))

    # Prediction of test data
    viewResults(test_ds, prediction_model)


if __name__ == "__main__":
    main()
