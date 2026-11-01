
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold

from keras.models import Sequential
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping


# =========================================================
# 1. LOAD DATASET
# =========================================================

print("1. Loading dataset...")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"

data = pd.read_csv(url, header=None)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

print("Dataset loaded successfully!")
print("Dataset shape:", X.shape)


# =========================================================
# 2. DATA PREPROCESSING
# =========================================================

print("\n2. Preparing dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Data preprocessing completed!")


# =========================================================
# 3. BUILD NEURAL NETWORK MODEL
# =========================================================

def build_model(unit_size=128, activation='relu'):

    model = Sequential([
        Input(shape=(57,)),

        Dense(units=unit_size, activation=activation),
        Dense(units=unit_size, activation=activation),

        Dense(units=1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# =========================================================
# EXPERIMENT 1: HIDDEN LAYER UNIT SIZES
# =========================================================

print("\n3. Experiment 1: Hidden Layer Unit Sizes")

unit_sizes = [32, 64, 128]

histories = {}
test_accuracies = {}
cv_accuracies = {}

kf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

for size in unit_sizes:

    print(f"\nTraining model with {size} hidden units...")

    model = build_model(
        unit_size=size,
        activation='relu'
    )

    early_stop = EarlyStopping(
        monitor='loss',
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        verbose=0,
        callbacks=[early_stop]
    )

    histories[size] = history.history['accuracy']

    _, test_acc = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    test_accuracies[size] = test_acc

    print(f"Test Accuracy: {test_acc:.4f}")

    # Cross Validation

    print(f"Calculating Cross Validation for {size} units...")

    fold_accs = []

    for train_idx, val_idx in kf.split(X, y):

        X_fold_train = X[train_idx]
        X_fold_val = X[val_idx]

        fold_scaler = StandardScaler()

        X_fold_train = fold_scaler.fit_transform(X_fold_train)
        X_fold_val = fold_scaler.transform(X_fold_val)

        cv_model = build_model(
            unit_size=size,
            activation='relu'
        )

        cv_model.fit(
            X_fold_train,
            y[train_idx],
            epochs=100,
            batch_size=32,
            verbose=0,
            callbacks=[EarlyStopping(
                monitor='loss',
                patience=10,
                restore_best_weights=True
            )]
        )

        _, val_acc = cv_model.evaluate(
            X_fold_val,
            y[val_idx],
            verbose=0
        )

        fold_accs.append(val_acc)

    cv_accuracies[size] = np.mean(fold_accs)

    print(f"Cross Validation Accuracy: {cv_accuracies[size]:.4f}")


# =========================================================
# EXPERIMENT 2: ACTIVATION FUNCTIONS
# =========================================================

print("\n4. Experiment 2: Activation Functions")

activations = [
    'relu',
    'tanh',
    'sigmoid',
    'softmax',
    'swish',
    'elu',
    'selu'
]

act_general_acc = {}

for act in activations:

    print(f"\nTesting activation function: {act}")

    model = build_model(
        unit_size=128,
        activation=act
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        verbose=0,
        callbacks=[EarlyStopping(
            monitor='loss',
            patience=10,
            restore_best_weights=True
        )]
    )

    train_acc = history.history['accuracy'][-1]

    _, test_acc = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    general_acc = (train_acc + test_acc) / 2

    act_general_acc[act] = general_acc

    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"General Accuracy: {general_acc:.4f}")


# =========================================================
# 5. PRINT EXPERIMENTAL RESULTS
# =========================================================

print("\n5. EXPERIMENTAL RESULTS")

print("\nHidden Layer Unit Sizes:")

for size in unit_sizes:

    print(
        f"Units: {size} | "
        f"Test Accuracy: {test_accuracies[size]:.4f} | "
        f"CV Accuracy: {cv_accuracies[size]:.4f}"
    )

print("\nActivation Functions:")

for act in activations:

    print(
        f"{act}: {act_general_acc[act]:.4f}"
    )


# =========================================================
# FIGURE 1: CONVERGENCE PLOT
# =========================================================

print("\n6. Generating Figure 1...")

plt.figure(figsize=(9, 5))

for size in unit_sizes:

    plt.plot(
        histories[size],
        label=f'Unit Size {size}'
    )

plt.title('Convergence Plot')
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy')

plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    'figure1_convergence.png',
    dpi=300,
    bbox_inches='tight'
)


# =========================================================
# FIGURE 2: TEST ACCURACY VS CROSS VALIDATION
# =========================================================

print("Generating Figure 2...")

x_labels = [str(s) for s in unit_sizes]
x = np.arange(len(x_labels))

width = 0.35

fig2, ax = plt.subplots(figsize=(9, 5))

rects1 = ax.bar(
    x - width / 2,
    [test_accuracies[s] for s in unit_sizes],
    width,
    label='Test Accuracy'
)

rects2 = ax.bar(
    x + width / 2,
    [cv_accuracies[s] for s in unit_sizes],
    width,
    label='Cross Validation Accuracy'
)

ax.set_ylabel('Accuracy')
ax.set_xlabel('Hidden Layer Unit Size')

ax.set_title('Test Accuracy vs Cross Validation Accuracy')

ax.set_xticks(x)
ax.set_xticklabels(x_labels)

ax.set_ylim(0, 1.05)

ax.legend()
ax.grid(axis='y', alpha=0.3)


for rect in rects1 + rects2:

    height = rect.get_height()

    ax.annotate(
        f'{height:.3f}',
        xy=(
            rect.get_x() + rect.get_width() / 2,
            height
        ),
        xytext=(0, 4),
        textcoords="offset points",
        ha='center',
        va='bottom'
    )

plt.tight_layout()

plt.savefig(
    'figure2_comparison.png',
    dpi=300,
    bbox_inches='tight'
)


# =========================================================
# FIGURE 3: ACTIVATION FUNCTIONS COMPARISON
# =========================================================

print("Generating Figure 3...")

plt.figure(figsize=(10, 5))

bars = plt.bar(
    activations,
    [act_general_acc[a] for a in activations]
)

plt.title('Comparison of Activation Functions')
plt.xlabel('Activation Function')
plt.ylabel('General Accuracy')

plt.ylim(0, 1.05)

plt.xticks(rotation=45)

for bar in bars:

    yval = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 0.005,
        f'{yval:.4f}',
        ha='center',
        va='bottom',
        fontsize=9
    )

plt.tight_layout()

plt.savefig(
    'figure3_activations.png',
    dpi=300,
    bbox_inches='tight'
)


# =========================================================
# 7. DISPLAY ALL THREE FIGURES
# =========================================================

print("\n7. All experiments completed successfully!")

print("\nThree figures have been saved:")
print("1. figure1_convergence.png")
print("2. figure2_comparison.png")
print("3. figure3_activations.png")

plt.show()
