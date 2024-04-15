import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, concatenate
from tensorflow.keras.layers import Average, Multiply, Add
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau, LearningRateScheduler
from datetime import datetime
import numpy as np


IMG_HEIGHT = 256
IMG_WIDTH = 256

# Data preparation
def encode_data(csv_path):
    df_init = pd.read_csv(csv_path)
    df_init = df_init.dropna()

    # Remove duplicates
    df_wo_duplicates = df_init.drop_duplicates(subset='lesion_id')
    df = df_wo_duplicates.drop(['mode', 'height', 'width', 'format', 'directory', 'file_name', 'lesion_id', 'image_id', 'dx', 'label'], axis=1)

    # Categorical data
    categorical_columns = ['dx_type', 'sex', 'localization']
    label_encoder = LabelEncoder()
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])

    # Numerical data
    numerical_columns = ['age']
    numerical_preprocessor = MinMaxScaler()
    df[numerical_columns] = numerical_preprocessor.fit_transform(df[numerical_columns])

    # Labels one-hot encoded
    labels = df_wo_duplicates.label.values.reshape(-1, 1)
    encoder = OneHotEncoder(categories='auto', sparse_output=False)
    one_hot_labels = encoder.fit_transform(labels)
    one_hot_labels_df = pd.DataFrame(one_hot_labels, columns=[f'class_{i}' for i in range(one_hot_labels.shape[1])])

    return df, one_hot_labels_df


def preprocess_img(img_path, resize=True):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img)

    if resize:
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])

    img /= 255.0
    return img

def get_img_meta_labels(dataset_dir, X, y):
    # Images
    img_paths = [os.path.join(dataset_dir, path) for path in X.filepath.values]
    img_dataset = tf.data.Dataset.from_tensor_slices(img_paths)
    img_dataset = img_dataset.map(preprocess_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Metadata
    metadata_dataset = tf.data.Dataset.from_tensor_slices(X[['dx_type', 'age', 'sex', 'localization']].values)

    # Labels
    labels_dataset = tf.data.Dataset.from_tensor_slices(y)

    return img_dataset, metadata_dataset, labels_dataset

def get_zip_dataset(img_dataset, metadata_dataset, labels_dataset, batch_size):
    dataset = tf.data.Dataset.zip((img_dataset, metadata_dataset))
    dataset = tf.data.Dataset.zip((dataset, labels_dataset))

    dataset = dataset.batch(batch_size)
    return dataset

def get_datasets(csv_path, batch_size, resize, num_classes, num_metadata_features):
    # Data encoding
    df, df_target_label = encode_data(csv_path)
    print(df.shape, df_target_label.shape)
    df.head()

    # Split train/val/test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(df, df_target_label, test_size=0.10, shuffle=True, stratify=df_target_label, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.10, shuffle=True, stratify=y_train_val, random_state=42)
    # X_train_val, X_test, y_train_val, y_test = train_test_split(df, df_target_label, test_size=0.10, shuffle=True, random_state=42)
    # X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.10, shuffle=True, random_state=42)


    train_img_dataset, train_metadata_dataset, train_labels = get_img_meta_labels(dataset_dir, X_train, y_train)
    val_img_dataset, val_metadata_dataset, val_labels = get_img_meta_labels(dataset_dir, X_val, y_val)
    test_img_dataset, test_metadata_dataset, test_labels = get_img_meta_labels(dataset_dir, X_test, y_test)

    train_dataset = get_zip_dataset(train_img_dataset, train_metadata_dataset, train_labels, batch_size)
    val_dataset = get_zip_dataset(val_img_dataset, val_metadata_dataset, val_labels, batch_size)
    test_dataset = get_zip_dataset(test_img_dataset, test_metadata_dataset, test_labels, batch_size)

    return train_dataset, val_dataset, test_dataset


# CNN to extract features from image
def get_cnn_model(num_classes):
    img_inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = Conv2D(32, (3, 3), activation='relu')(img_inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dense(512, activation="relu")(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    x= Dense(64, activation="relu")(x)
    output = Dense(num_classes, activation='softmax')(x)

    return img_inputs, output, x

# Dense layers for text metadata
def get_dense_model(num_classes, num_metadata_features):
    metadata_inputs = Input(shape=(num_metadata_features, ))
    meta = Dense(128, activation="relu")(metadata_inputs)
    meta = Dropout(0.3)(meta)
    meta = Dense(64, activation="relu")(meta)
    meta = Dropout(0.3)(meta)
    meta = Dense(32, activation="relu")(meta)
    meta = Dropout(0.3)(meta)
    meta = Dense(16, activation="relu")(meta)
    output = Dense(num_classes, activation='softmax')(meta)

    return metadata_inputs, output, meta


def combine_models(img_inputs, cnn_outputs, cnn_x, metadata_inputs, dense_outputs, meta_x, num_classes):
    # Average the outputs of the two models
    combined_outputs = concatenate([cnn_x, meta_x]) #Ctrl KC/KU

    # Additional dense layers for classification
    x = Dense(256, activation="relu")(combined_outputs)
    x = Dense(128, activation="relu")(x)
    x = Dense(num_classes, activation='softmax')(x)

    # Define the combined model
    combined_model = Model(inputs=[img_inputs, metadata_inputs], outputs=x)
    combined_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    print(combined_model.summary())
    return combined_model


def combine_weighted_models(img_inputs, cnn_outputs, metadata_inputs, dense_outputs, num_classes, cnn_weight=0.5, dense_weight=0.5):
    # Multiply each output with its respective weight
    cnn_weighted = Multiply()([cnn_outputs, cnn_weight])
    dense_weighted = Multiply()([dense_outputs, dense_weight])

    # Combine the weighted outputs
    combined_outputs = Add()([cnn_weighted, dense_weighted])

    # Additional dense layers for classification
    x = Dense(256, activation="relu")(combined_outputs)
    x = Dense(128, activation="relu")(x)
    x = Dense(num_classes, activation='softmax')(x)

    # Define the combined model
    combined_model = Model(inputs=[img_inputs, metadata_inputs], outputs=x)
    combined_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return combined_model

def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr  # Keep initial learning rate for the first 10 epochs
    else:
        return lr * tf.math.exp(-0.1)  # Exponential decay after the 10th epoch


# Define a custom callback to plot metrics during training
class PlotMetricsCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.losses = []
        self.val_losses = []
        self.accs = []
        self.val_accs = []
        self.lrs = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs['loss'])
        self.val_losses.append(logs['val_loss'])
        self.accs.append(logs['categorical_accuracy'])
        self.val_accs.append(logs['val_categorical_accuracy'])
        self.lrs.append(self.model.optimizer.lr.numpy())  # Get current learning rate
        epochs = np.arange(1, epoch + 2)

        # Plot loss and accuracy
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.losses, label='Training Loss')
        plt.plot(epochs, self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.accs, label='Training Accuracy')
        plt.plot(epochs, self.val_accs, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

def train_model(train_dataset, val_dataset, num_classes, num_metadata_features, nb_epochs, batch_size, dataset_dir):
    # Load the model and compile it
    img_inputs, cnn_outputs, cnn_x = get_cnn_model(num_classes)
    metadata_inputs, dense_outputs, meta_x = get_dense_model(num_classes, num_metadata_features)
    model = combine_models(img_inputs, cnn_outputs, cnn_x, metadata_inputs, dense_outputs, meta_x, num_classes)
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['categorical_accuracy'])


    # Define callbacks
    checkpoint_filepath = os.path.join(dataset_dir, 'model2_checkpoint.h5')
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        save_freq="epoch",
        verbose=1
    )
    # plot_callback = PlotMetricsCallback()
    early_stopping_callback = EarlyStopping(monitor='val_loss',
                        patience=8,
                        mode='min',
                        restore_best_weights=True, verbose=1)

    csv_logger_callback = CSVLogger(os.path.join(dataset_dir, 'model2_training_history.csv'))
    reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6,  mode="min", verbose=1)
    lr_scheduler_callback = LearningRateScheduler(lr_scheduler)

    # Train the model with callbacks
    history = model.fit(train_dataset,
                        epochs=nb_epochs,
                        batch_size=batch_size,
                        validation_data=val_dataset,
                        callbacks=[model_checkpoint_callback, early_stopping_callback, csv_logger_callback, reduce_lr_callback, lr_scheduler_callback])

    return model, history


if __name__=="__main__":
    dataset_dir = "/Users/marguerite/workspace_DS/project_CV/"
    csv_path = os.path.join(dataset_dir, "all_paths_metadata_df.csv")
    batch_size = 32
    resize = True
    num_classes = 7
    num_metadata_features = 4
    nb_epochs = 30
    train_dataset, val_dataset, test_dataset = get_datasets(csv_path, batch_size, resize, num_classes, num_metadata_features)

    model, history = train_model(train_dataset, val_dataset, num_classes, num_metadata_features, nb_epochs, batch_size, dataset_dir)

#TODO : train CNN model (input: image, target : pathologie)
#TODO : train dense (input: metadata, target : pathologie)
#TODO: train the entire model
#TODO : freeze CNN model et metadata model
# use model.layers : boucle for pour en mettre certaines en trainable=False
