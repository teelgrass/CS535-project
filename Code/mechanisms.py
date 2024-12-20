# Import necessary libraries
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Layer

# -------------------------------
# Radial Basis Mapping Layer
# -------------------------------

class RadialBasisMapping(Layer):
    """
    Radial Basis Mapping Layer using Euclidean distance.
    """
    def __init__(self, output_dim, **kwargs):
        super(RadialBasisMapping, self).__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.input_dim = int(np.prod(input_shape[1:]))
        self.centers = self.add_weight(
            shape=(self.output_dim, self.input_dim),
            initializer="random_normal",
            trainable=True,
            name="centers"
        )
        self.beta = self.add_weight(
            shape=(self.output_dim,),
            initializer="ones",
            trainable=True,
            name="beta"
        )

    def call(self, inputs):
        inputs_flat = tf.reshape(inputs, [tf.shape(inputs)[0], -1])
        diff = tf.expand_dims(inputs_flat, axis=1) - self.centers
        distance = tf.reduce_sum(tf.square(diff), axis=-1)
        return tf.exp(-self.beta * distance)

# -------------------------------
# Lightweight CNN Model
# -------------------------------

def build_lightweight_model(input_shape, num_classes):
    """
    Builds a lightweight CNN model with simplified Radial Basis Mapping.
    """
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Flatten(),
        RadialBasisMapping(output_dim=16),  # Reduced output dimension for faster processing
        Dense(64, activation='relu'),
        Dropout(0.3),  # Reduced dropout rate for faster training
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# -------------------------------
# Simplified FGSM Attack
# -------------------------------

@tf.function(reduce_retracing=True)
def fgsm_attack(model, image, label, epsilon):
    """
    FGSM (Fast Gradient Sign Method) attack implementation.
    """
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = tf.keras.losses.categorical_crossentropy(label, prediction)
        gradient = tape.gradient(loss, image)
    signed_grad = tf.sign(gradient)
    adversarial_image = image + epsilon * signed_grad
    return tf.clip_by_value(adversarial_image, 0, 1)

# -------------------------------
# Training with Adversarial Data (Optimized)
# -------------------------------

def train_with_adversarial_data(model, train_generator, attack_fn, epsilon, epochs=3, steps_per_epoch=None):
    """
    Incorporates adversarial training with a reduced number of epochs.
    """
    if steps_per_epoch is None:
        steps_per_epoch = np.ceil(train_generator.samples / train_generator.batch_size).astype(int)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        batch_count = 0
        
        for images, labels in train_generator:
            if batch_count >= steps_per_epoch:
                break  # Stop after the expected number of batches
            
            adversarial_images = attack_fn(model, images, labels, epsilon)
            augmented_images = np.concatenate([images, adversarial_images], axis=0)
            augmented_labels = np.concatenate([labels, labels], axis=0)
            model.train_on_batch(augmented_images, augmented_labels)
            batch_count += 1
            print(f"  Processed batch {batch_count}/{steps_per_epoch}")

        print(f"Epoch {epoch + 1} completed.")
    print("Adversarial training completed.")

# -------------------------------
# Evaluation Pipeline
# -------------------------------

def evaluate_lightweight_defenses(model, validation_generator, attack_fn, epsilon_values):
    """
    Evaluates the model's performance on clean and adversarial examples with different epsilon values.
    """
    print("Evaluating model performance...")
    results = {}

    # Evaluate clean accuracy
    clean_loss, clean_acc = model.evaluate(validation_generator, verbose=0)
    print(f"Clean Accuracy: {clean_acc * 100:.2f}%")
    results["Clean Accuracy"] = clean_acc * 100

    # Evaluate adversarial accuracy for each epsilon
    for epsilon in epsilon_values:
        print(f"Evaluating for epsilon={epsilon}...")
        adversarial_examples, adversarial_labels = [], []
        for images, labels in validation_generator:
            adversarial_batch = attack_fn(model, images, labels, epsilon)
            adversarial_examples.append(adversarial_batch)
            adversarial_labels.append(labels)

        adversarial_dataset = tf.data.Dataset.from_tensor_slices(
            (np.vstack(adversarial_examples), np.vstack(adversarial_labels))
        ).batch(32)

        adv_loss, adv_acc = model.evaluate(adversarial_dataset, verbose=0)
        print(f"  Adversarial Accuracy (ε={epsilon}): {adv_acc * 100:.2f}%")
        results[f"Adversarial Accuracy (ε={epsilon})"] = adv_acc * 100

    return results

# -------------------------------
# Main Execution (Optimized)
# -------------------------------

# Data Augmentation and Loading
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,  # Reduced rotation range
    width_shift_range=0.05,  # Reduced shift range
    height_shift_range=0.05,  # Reduced shift range
    horizontal_flip=True
)
train_generator = datagen.flow_from_directory(
    'output_dataset/train',
    target_size=(128, 128),
    batch_size=64,  # Reduced batch size for faster training
    class_mode='categorical'
)
validation_generator = datagen.flow_from_directory(
    'output_dataset/validation',
    target_size=(128, 128),
    batch_size=64,  # Reduced batch size for validation
    class_mode='categorical'
)

# Build and train the model
input_shape = (128, 128, 3)
num_classes = 8
lightweight_model = build_lightweight_model(input_shape, num_classes)
train_with_adversarial_data(lightweight_model, train_generator, fgsm_attack, epsilon=0.01, epochs=4)

# Save the trained model
lightweight_model.save('lightweight_model.h5')
print("Model saved to lightweight_model.h5")

# Evaluate the model
epsilon_values = [0.01, 0.02, 0.03]
results = evaluate_lightweight_defenses(lightweight_model, validation_generator, fgsm_attack, epsilon_values)
print("Evaluation Results:", results)
