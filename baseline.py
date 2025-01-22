import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import json


def create_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model


#preprocess
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0

x_member, y_member = x_train[:25000], y_train[:25000]
x_non_member, y_non_member = x_test[:10000], y_test[:10000]
x_val, y_val = x_train[25000:30000], y_train[25000:30000]

#train
baseline_model = create_cnn()
baseline_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
baseline_model.fit(x_member, y_member, epochs=20, validation_data=(x_val, y_val), batch_size=256)

def evaluate_mia(model, member_data, non_member_data, threshold):
    member_probs = model.predict(member_data)
    non_member_probs = model.predict(non_member_data)

    member_conf = np.max(member_probs, axis=1)
    non_member_conf = np.max(non_member_probs, axis=1)

    true_labels = np.concatenate([np.ones(len(member_data)), np.zeros(len(non_member_data))])
    pred_labels = np.concatenate([
        (member_conf > threshold).astype(int),
        (non_member_conf > threshold).astype(int)
    ])

    acc = float(np.mean(true_labels == pred_labels))  # Convert to Python float
    roc_auc = float(tf.keras.metrics.AUC()(true_labels, np.concatenate([member_conf, non_member_conf])).numpy())  # Convert to Python float
    return acc, roc_auc

thresholds = np.linspace(0.5, 0.99, 20)
baseline_results = []

for th in thresholds:
    acc, roc_auc = evaluate_mia(baseline_model, x_member, x_non_member, th)
    baseline_results.append({
        "threshold": float(th),  # Convert to Python float
        "accuracy": acc,
        "roc_auc": roc_auc
    })

with open("results_baseline.json", "w") as f:
    json.dump(baseline_results, f)