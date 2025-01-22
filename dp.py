import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import json

# custom optimizer because of tensorflow import difficulties/incompatibilities
class CustomDPSGDOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(
        self,
        learning_rate=0.15,
        l2_norm_clip=1.0,
        noise_multiplier=0.5,
        num_microbatches=256,
        name="CustomDPSGD",
        **kwargs
    ):
        super().__init__(learning_rate=learning_rate, name=name, **kwargs)
        self._l2_norm_clip = l2_norm_clip
        self._noise_multiplier = noise_multiplier
        self._num_microbatches = num_microbatches
        self._lr = tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate, decay_steps=1000, decay_rate=0.9
        )

    def update_step(self, gradient, variable, learning_rate):
        lr_t = self._lr(self.iterations)
        
        # Clip gradients
        grad_norm = tf.norm(gradient)
        gradient = gradient * tf.minimum(1.0, self._l2_norm_clip / (grad_norm + 1e-10))
        
        # Add noise
        noise_stddev = self._l2_norm_clip * self._noise_multiplier
        noise = tf.random.normal(shape=gradient.shape, stddev=noise_stddev)
        noised_grad = gradient + noise
        
        # Apply gradients using provided learning rate
        variable.assign_sub(learning_rate * noised_grad)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "accumulator")

    def _resource_apply_dense(self, grad, var, apply_state=None):
        learning_rate = self._lr(self.iterations)
        return self.update_step(grad, var, learning_rate)

    def get_config(self):
        config = super().get_config()
        config.update({
            "learning_rate": self._learning_rate,
            "l2_norm_clip": self._l2_norm_clip,
            "noise_multiplier": self._noise_multiplier,
            "num_microbatches": self._num_microbatches,
        })
        return config

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



#preprocess
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0

x_member, y_member = x_train[:25000], y_train[:25000]
x_non_member, y_non_member = x_test[:10000], y_test[:10000]
x_val, y_val = x_train[25000:30000], y_train[25000:30000]

#train
dp_optimizer = CustomDPSGDOptimizer(learning_rate=0.15, l2_norm_clip=1.0, noise_multiplier=0.5, num_microbatches=256)
dp_model = create_cnn()
dp_model.compile(optimizer=dp_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
dp_model.fit(x_member, y_member, epochs=20, validation_data=(x_val, y_val), batch_size=256)

thresholds = np.linspace(0.5, 0.99, 20)
dp_results = []

for th in thresholds:
    acc, roc_auc = evaluate_mia(dp_model, x_member, x_non_member, th)
    dp_results.append({
        "threshold": float(th),  # Convert to Python float
        "accuracy": acc,
        "roc_auc": roc_auc
    })

with open("results_dp.json", "w") as f:
    json.dump(dp_results, f)