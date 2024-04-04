import os
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, concatenate


IMG_HEIGHT = 128
IMG_WIDTH = 128

# Data preparation
def encode_data(csv_path):
    df_init = pd.read_csv(csv_path)
    df_target_label = df_init.label
    df = df_init.drop(['mode', 'height', 'width', 'format', 'directory', 'file_name', 'lesion_id', 'image_id', 'dx', 'label'], axis=1)

    # Categorical data
    categorical_columns = ['dx_type', 'sex', 'localization']
    label_encoder = LabelEncoder()
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])

    # Numerical data
    numerical_columns = ['age']
    numerical_preprocessor = MinMaxScaler()
    df[numerical_columns] = numerical_preprocessor.fit_transform(df[numerical_columns])

    return df, df_target_label

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
    labels_dataset = tf.data.Dataset.from_tensor_slices(y.values)

    return img_dataset, metadata_dataset, labels_dataset

def get_zip_dataset(img_dataset, metadata_dataset, labels_dataset, batch_size):
        dataset = tf.data.Dataset.zip((img_dataset, metadata_dataset))
        dataset = tf.data.Dataset.zip((dataset, labels_dataset))

        dataset = dataset.batch(batch_size)
        return dataset

def get_model(num_classes, num_metadata_features):
    # CNN to extract features from image
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
    
    # Dense layers for text metadata
    metadata_inputs = Input(shape=(num_metadata_features, ))
    meta = Dense(128, activation="relu")(metadata_inputs)
    meta = Dense(128, activation="relu")(meta)
    meta = Dense(128, activation="relu")(meta)
    meta = Dense(128, activation="relu")(meta)
    
    # Combine image and text features + classification 
    combined = concatenate([x, meta])
    z = Dense(64, activation='relu')(combined)
    z = Dense(32, activation='relu')(z)
    output = Dense(num_classes, activation='softmax')(z)
    
    # Create the model
    model = Model(inputs=[img_inputs, metadata_inputs], outputs=output)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def train_op():
    dataset_dir = "/Users/marguerite/workspace_DS/project_CV/"
    batch_size = 32
    resize = True
    num_classes = 15
    num_metadata_features = 4
    nb_epochs = 2

    # Data encoding
    csv_path = os.path.join(dataset_dir, "all_paths_metadata_df.csv")
    df, df_target_label = encode_data(csv_path)
    df.head()

    # Split train/val/test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(df, df_target_label, test_size=0.2, shuffle=True, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15, shuffle=True, random_state=42)

    train_img_dataset, train_metadata_dataset, train_labels = get_img_meta_labels(dataset_dir, X_train, y_train)
    val_img_dataset, val_metadata_dataset, val_labels = get_img_meta_labels(dataset_dir, X_val, y_val)
    test_img_dataset, test_metadata_dataset, test_labels = get_img_meta_labels(dataset_dir, X_test, y_test)

    train_dataset = get_zip_dataset(train_img_dataset, train_metadata_dataset, train_labels, batch_size)
    val_dataset = get_zip_dataset(val_img_dataset, val_metadata_dataset, val_labels, batch_size)
    test_dataset = get_zip_dataset(test_img_dataset, test_metadata_dataset, test_labels, batch_size)

    model = get_model(num_classes, num_metadata_features)
    model.fit(train_dataset, epochs=nb_epochs, batch_size=batch_size, validation_data=val_dataset)

    #TODO : add callbacks, check gradients 
    

if __name__=="__main__":
     train_op()