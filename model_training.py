import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

DATA_PATH = 'dataset_pose'
ACTIONS = ['normal', 'anomalie']
SEQUENCE_LENGTH = 30

X, y = [], []

print("Chargement...")


def process_sequence(seq):

    seq = seq.reshape(seq.shape[0], -1, 2)

    seq = seq - seq[:, :1, :]

    seq = seq.reshape(seq.shape[0], -1)

    velocity = seq[1:] - seq[:-1]
    velocity = np.vstack([velocity, np.zeros((1, seq.shape[1]))])

    seq = np.concatenate([seq, velocity], axis=1)

    return seq


for label_idx, action in enumerate(ACTIONS):

    path = os.path.join(DATA_PATH, action)

    if not os.path.exists(path):
        continue

    for f in os.listdir(path):

        if f.endswith('.npy'):

            seq = np.load(os.path.join(path, f))

            if seq.shape == (SEQUENCE_LENGTH, 34):

                seq = process_sequence(seq)
                X.append(seq)
                y.append(label_idx)

X = np.array(X)
y = np.array(y)

print("Dataset:", X.shape)


mean = X.mean()
std = X.std()

X = (X - mean) / std


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


def count_classes(labels):
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))

train_counts = count_classes(y_train)
test_counts = count_classes(y_test)
total_counts = count_classes(y)

print("\n Répartition des données :")
print("Train :", train_counts)
print("Validation :", test_counts)
print("Total :", total_counts)

velocity_data = X[:, :, 34:]

velocity_normal = velocity_data[y == 0]
velocity_anomalie = velocity_data[y == 1]

def compute_stats(data):
    return {
        "mean": np.mean(data),
        "std": np.std(data),
        "max": np.max(data),
        "p95": np.percentile(data, 95)
    }

stats_normal = compute_stats(velocity_normal)
stats_anomalie = compute_stats(velocity_anomalie)

print("\n Statistiques vélocité :")
print("Normal :", stats_normal)
print("Anomalie :", stats_anomalie)


weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: weights[i] for i in range(len(weights))}


model = Sequential([

    Input(shape=(SEQUENCE_LENGTH, 68)),  

    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.3),

    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.3),

    LSTM(64),

    Dense(128, activation='relu'),
    Dropout(0.4),

    Dense(1, activation='sigmoid')

])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()


early_stop = EarlyStopping(patience=10, restore_best_weights=True)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=200,
    batch_size=16,
    callbacks=[early_stop],
    class_weight=class_weights
)
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Loss entraînement', color='blue')
plt.plot(history.history['val_loss'], label='Loss validation', color='orange')
plt.plot(history.history['accuracy'], label='Accuracy entraînement', color='green')
plt.plot(history.history['val_accuracy'], label='Accuracy validation', color='red')

plt.title("Évolution de la perte (Loss) et de la précision (Accuracy)")
plt.xlabel("Epochs")
plt.ylabel("Valeur")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

seq_normal = X[y == 0][0] 
positions = seq_normal[:SEQUENCE_LENGTH, :34]  
positions = positions.reshape(SEQUENCE_LENGTH, -1, 2)

vel = positions[1:] - positions[:-1]
vel_magnitude = np.linalg.norm(vel, axis=2)
vel_magnitude = np.vstack([vel_magnitude, np.zeros((1, vel_magnitude.shape[1]))])

plt.figure(figsize=(10, 4))
plt.plot(vel_magnitude)
plt.title("Évolution de la vélocité ∆Pt pour une séquence normale")
plt.xlabel("Frame")
plt.ylabel("Vélocité (magnitude)")
plt.show() 



SEQUENCE_LENGTH = 30

seq_anomalie = X[y == 1][0]  
positions = seq_anomalie[:SEQUENCE_LENGTH, :34] 
positions = positions.reshape(SEQUENCE_LENGTH, -1, 2)

vel = positions[1:] - positions[:-1] 
vel_magnitude = np.linalg.norm(vel, axis=2)  
vel_magnitude = np.vstack([vel_magnitude, np.zeros((1, vel_magnitude.shape[1]))])  


plt.figure(figsize=(10, 4))
plt.plot(vel_magnitude)
plt.title("Évolution de la vélocité ∆Pt pour une séquence anormale (chute)")
plt.xlabel("Frame")
plt.ylabel("Vélocité (magnitude)")
plt.show()


os.makedirs("models", exist_ok=True)
model.save("models/action_model_final.keras")

np.save("models/mean.npy", mean)
np.save("models/std.npy", std)

print("Modèle sauvegardé")


print("\nÉvaluation du modèle...")

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    target_names=ACTIONS
)

print(f"\n Accuracy : {accuracy*100:.2f}%")
print(f"Loss : {loss:.4f}")

print("\n Confusion Matrix :")
print(cm)

print("\n Classification Report :")
print(report)



with open("models/metrics.txt", "w") as f:

    f.write(f"Accuracy: {accuracy*100:.2f}%\n")
    f.write(f"Loss: {loss:.4f}\n\n")

    f.write("Confusion Matrix:\n")
    f.write(str(cm))

    f.write("\n\nClassification Report:\n")
    f.write(report)

print("Métriques sauvegardées dans models/metrics.txt")