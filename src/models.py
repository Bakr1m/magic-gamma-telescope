"""Classifiers: thesis MLP (verbatim) + classical baselines."""
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def build_mlp(num_nodes: int, dropout_prob: float, lr: float, n_features: int = 10):
    """2x [Dense -> Dropout] + sigmoid head (thesis cell 23)."""
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_features,)),
            tf.keras.layers.Dense(num_nodes, activation="relu"),
            tf.keras.layers.Dropout(dropout_prob),
            tf.keras.layers.Dense(num_nodes, activation="relu"),
            tf.keras.layers.Dropout(dropout_prob),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_mlp(X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs):
    model = build_mlp(num_nodes, dropout_prob, lr, n_features=X_train.shape[1])
    history = model.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size,
        validation_split=0.2, verbose=0,
    )
    return model, history


def build_baseline(name: str):
    """Classical baselines from the notebook (default hyperparameters)."""
    baselines = {
        "knn": KNeighborsClassifier(n_neighbors=5),
        "naive_bayes": GaussianNB(),
        "logistic_regression": LogisticRegression(max_iter=1000),
        "svm": SVC(),
    }
    if name not in baselines:
        raise ValueError(f"unknown baseline {name!r}; choose from {sorted(baselines)}")
    return baselines[name]
