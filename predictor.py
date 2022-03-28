from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
from glob import glob

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

np.random.seed(42)
tf.random.set_seed(42)
base_path = "../data"
base_image_path = os.path.join(base_path,"full_data_org")


def loadData():
    lines_words = []
    words_path = os.path.join(base_path,"words.txt")
    words = open(words_path, "r",encoding="utf-8").readlines()
    for line in words:
        if line[0] == "#":
            continue
        if line.split(" ")[1] != "err":
            lines_words.append(line.strip())

    np.random.shuffle(lines_words)

    split_idx = int(0.9 * len(lines_words))
    train_data = lines_words[:split_idx]
    test_data = lines_words[split_idx:]

    val_split_idx = int(0.5 * len(test_data))
    validation_data = test_data[:val_split_idx]
    test_data = test_data[val_split_idx:]

    assert len(lines_words) == len(train_data) + len(validation_data) + len(
        test_data
    )
    return train_data, test_data, validation_data

def loadSamples():
    base_path = "../"
    lines_words = []
    words_path = os.path.join(base_path,"samples.txt")
    words = open(words_path, "r",encoding="utf-8").readlines()
    for line in words:
        lines_words.append(line.strip())

    return lines_words


def get_image_paths_and_labels(data,base_image_path=base_image_path):
    paths = []
    labels = []
    for (i, file_line) in enumerate(data):
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
    for batch in test_ds.take(5):
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

            title = f"Prediction: {pred_texts[i]} \n gt_text:   {gt_texts[i]}"
            ax[i // 4, i % 4].imshow(img, cmap="gray")
            ax[i // 4, i % 4].set_title(title)
            ax[i // 4, i % 4].axis("off")

    plt.show()



def main():
    global max_length, char_to_num, num_to_char, padding_token

    batch_size = 64
    padding_token = 99

    model_name = "model_encode_fix"
    model_folder = "../saved_models"

    train_data, test_data, validation_data = loadData()
    sample_data = loadSamples()

    train_img_paths, train_labels = get_image_paths_and_labels(train_data)
    validation_img_paths, validation_labels = get_image_paths_and_labels(validation_data)
    test_img_paths, test_labels = get_image_paths_and_labels(test_data)
    samples_paths, sample_labels = get_image_paths_and_labels(sample_data,base_image_path="../")


    max_length, characters = get_max_length_and_chars(train_labels)

    print(characters)

    # Mapping characters to integers.
    char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)

    # Mapping integers back to original characters.
    num_to_char = StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )

    train_ds = prepare_dataset(train_img_paths, train_labels, batch_size)
    validation_ds = prepare_dataset(validation_img_paths, validation_labels, batch_size)
    test_ds = prepare_dataset(test_img_paths, test_labels, batch_size)
    sample_ds = prepare_dataset(samples_paths, sample_labels, 16)


    # Get model
    model = keras.models.load_model(os.path.join(model_folder, model_name))

    # Get the prediction model by extracting layers till the output layer
    prediction_model = keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="reshape3").output 
    )

    # Compute accuracy
    # acc = accuracy(validation_ds,prediction_model)
    # print(f"validation accuracy = {acc}" )
    # acc = accuracy(test_ds,prediction_model)
    # print(f"test accuracy = {acc}" )



    # Prediction of test data
    viewResults(sample_ds, prediction_model)

    acc = accuracy(sample_ds,prediction_model)
    print(f"sample accuracy = {acc}" )

if __name__ == "__main__":
    main()
