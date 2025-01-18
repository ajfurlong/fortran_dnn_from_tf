import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import h5py
import time

def generate_data(num_samples=5000):
    """Generate data for a transformed nonlinear relationship with three variables."""
    tf.keras.utils.set_random_seed(1234)

    # Generate x1, x2, and x3 values
    x1 = np.random.uniform(0, 10, num_samples)
    x2 = np.random.uniform(1, 5, num_samples)
    x3 = np.random.uniform(-3, 3, num_samples)

    # Define a transformed nonlinear relationship with three inputs
    y = (2 * x1**2 - x2 * x3 + 4 * x3**2 - 3 * x1 * x2 + 7) + 0.1 * np.random.randn(num_samples)

    # Combine x1, x2, and x3 into a single input array
    x = np.column_stack((x1, x2, x3))
    y = y.reshape(-1, 1)

    # Shuffle the data
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    return x, y

def train_model(x_train, y_train, epochs=500, batch_size=32):
    """Create and fit the two-layer DNN model."""
    # Define the model with two input features
    model = keras.Sequential([
        layers.Dense(17, activation='relu', input_shape=(x_train.shape[1],)),
        layers.Dense(16, activation='elu'),
        layers.Dense(15, activation='tanh'),
        layers.Dense(14, activation='relu'),
        layers.Dense(1)
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    
    # Train the model
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    
    return model

def compute_metrics(y_true, y_pred, inference_time):
    """Compute and return various regression metrics."""
    relative_error = np.where(y_true != 0, 100 * np.abs((y_true - y_pred) / y_true), 0)
    absolute_error = np.abs(y_pred - y_true)
    ferr_above_10 = np.sum(relative_error > 10.0)
    rrmse = np.sqrt(np.mean(((y_pred - y_true) / y_true) ** 2))

    metrics = {
        "MAE": np.mean(absolute_error),
        "Max AE": np.max(absolute_error),
        "Min AE": np.min(absolute_error),
        "STD AE": np.std(absolute_error),
        "MAPE": np.mean(relative_error),
        "Max APE": np.max(relative_error),
        "Min APE": np.min(relative_error),
        "STD APE": np.std(relative_error),
        "rRMSE (%)": rrmse * 100,
        "Ferr > 10% (%)": 100 * ferr_above_10 / len(relative_error),
        "R^2": r2_score(y_true, y_pred),
        "Inference Time (s)": inference_time
    }
    return metrics

# Save data and model
def save_data_and_model(x_test_rescaled, y_test_rescaled, y_pred_rescaled, model,
                        x_scaler, y_scaler, time_elapsed,
                        data_file='../data/nonlinear_regression_data_test.h5',
                        model_file='../models/nonlinear_regression_model_tf.h5',
                        metadata_file='../models/nonlinear_regression_metadata_tf.h5'):
    """Save the test data and model."""
    model.save(model_file)
    print(f"Model has been saved to {model_file}")

    with h5py.File(metadata_file, 'w') as f:
        grp1 = f.create_group("scaler")
        grp1.create_dataset("x_mean", data=x_scaler.mean_)
        grp1.create_dataset("y_mean", data=y_scaler.mean_)
        grp1.create_dataset("x_std", data=x_scaler.scale_)
        grp1.create_dataset("y_std", data=y_scaler.scale_)
    print(f"Model metadata has been saved to {metadata_file}")

    test_data = np.hstack((x_test_rescaled, y_test_rescaled, y_pred_rescaled))
    headers = ["input1", "input2", "input3", "output_true", "output_pred"]

    with h5py.File(data_file, 'w') as f:
        for i, header in enumerate(headers):
            f.create_dataset(header, data=test_data[:, i])
        f.create_dataset("inference_time", data=time_elapsed)
    print(f"Data has been saved to {data_file}")

def main():
    # Generate data
    x, y = generate_data()
    
    # Split the data into training and test sets
    split_index = int(len(x) * 0.8)
    x_train, x_test = x[:split_index], x[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # Standardize the input data
    x_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_train)
    x_test = x_scaler.transform(x_test)

    # Standardize the output data
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)

    # Train the model
    model = train_model(x_train, y_train)

    # For timing statistics
    start_time = time.time()

    # Make predictions and inverse transform
    y_pred = model.predict(x_test)

    end_time = time.time()

    x_test_rescaled = x_scaler.inverse_transform(x_test)
    y_pred_rescaled = y_scaler.inverse_transform(y_pred)
    y_test_rescaled = y_scaler.inverse_transform(y_test)

    # Compute metrics
    metrics = compute_metrics(y_test_rescaled, y_pred_rescaled, end_time - start_time)

    # Create DataFrame for metrics
    metrics_df = pd.DataFrame(metrics, index=['Values']).T

    # Print metrics table
    print(metrics_df)

    # Save results table
    with open('results_nonlinear_regression_tf_dnn.txt', 'w') as f:
        f.write(metrics_df.to_string(float_format='%.6f'))

    # Print the means and standard deviations used for standardization
    print("Input means:", x_scaler.mean_)
    print("Output mean:", y_scaler.mean_)
    print("Input standard deviations:", x_scaler.scale_)
    print("Output standard deviation:", y_scaler.scale_)
    
    # Save the test data and model
    save_data_and_model(x_test_rescaled, y_test_rescaled, y_pred_rescaled,
                        model, x_scaler, y_scaler, end_time - start_time)

if __name__ == "__main__":
    main()