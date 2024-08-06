# Using Pre-Trained TensorFlow DNNs in Fortran

The purpose of this project is to provide a straightforward yet very accurate framework for using deep neural networks (DNNs) previously trained with TensorFlow to make predictions in a Fortran project. This approach allows Fortran programs to leverage the predictive power of TensorFlow-trained models without requiring complex integration or dependencies beyond HDF5 for data handling. There are a few other projects out there that focus on training and creating DNNs within Fortran, but this project simply does one thing (pretty well) that is easy to understand and implement.

## Project Overview

This project offers a minimalistic solution for incorporating TensorFlow-trained DNNs into Fortran applications. The project also includes data processing routines that facilitate benchmarking against original TensorFlow predictions to verify accuracy and consistency. Users are responsible for defining the network architecture manually, but the instructions provided make this process manageable and hopefully intuitive. The system is designed to process large HDF5 files for input and output data.

## Prerequisites

- **Fortran Compiler**: Ensure you have a Fortran compiler installed (e.g., `gfortran`).
- **HDF5 Library**: Required for handling HDF5 files. Install via package manager or from [HDF5 Group](https://www.hdfgroup.org/downloads/hdf5/).
- **Python**: Required for running the model conversion script.
- **TensorFlow**: Install via `pip install tensorflow` for converting models.

## Project Structure

- **bin/**: Directory for storing compiled executables.
- **src/**: Contains Fortran source files, including modules and the main program.
- **obj/**: Directory for storing compiled object and module files.
- **example/**: Contains example datasets and test cases.
- **models/**: Directory for storing model weights and biases in text format.
- **data/**: Directory for storing input and output datasets.

## Usage: Example

The included example covers the entire process from training the DNN on the TensorFlow side to running inference within Fortran. The example directory includes the data, model, and source script for a DNN learning a modified sinusoid.

### Step 1: Train the DNN using TensorFlow

The sinusoid.py script should already be configured correctly, so just verify that you have the correct packages installed and run it. This script first generates 5,000 points with a small amount stochastic noise (reproducible with the random seed being set). The data is shuffled, standardized (to mu=0 and sigma=1 to reduce bias from magnitude), and then 80% is used to train the model with 20% being held out for testing. 

The 1,000 test points (inputs, true outputs, and the TensorFlow model's predictions) are exported into an HDF5 file within the example/data/ directory. The model is saved as an HDF5 to example/model/ in a legacy format, but is still very effective (see enhancements section).

In the output of this script, there will be some performance statistics. Directly under these, are the input/output means and standard deviations used during the standardization process. Take note of these. They will have to be manually added in the main.f90 script so the Fortran implementation can understand how to standardize and unstandardize the new inputs and outputs from/to their physical dimensions.

### Step 2: Convert the TensorFlow Model

The dnn_module is designed to read the weights and biases of the TensorFlow model from a set of text files directly taken from the HDF5 file exported in the previous step. The script located in fortran_dnn_from_tf/models/ called convert_hdf5_txt.py performs this exact task. Run it in example/model and it will spit out a sinusoid_model_decomposed directory containing each layer's weights and biases in the same format as the HDF5 structure.

      python convert_hdf5_txt.py path/to/your_model.h5

### Step 3: Modify dnn_module.f90

This is where things get just a bit ugly. The first thing you will need to do is verify the network_depth parameter, which is equal to the number of hidden layers plus the output layer. The input_size parameter will then need to be verified, which is the number of input parameters you will provide the network per prediction. Then, the user must define each layer's number of neurons (units) by adding or subtracting inputX_size definitions and modifying their values.

In load_weights(), the only thing that needs to be changed is each layer's array allocations, where X is the unique layer number:

    ! Layer X
    allocate(network(X)%weights(layerX_size, input_size))
    allocate(network(X)%biases(layerX_size))

Now moving along to predict(), you will need to modify the daisy-chain of layers and how the information passes through them. The basic block is located below, which consists of allocating space in a temporary variable layer_output, performing the matrix multiplication with the weights and biases, and then applying the specified activation function. There are several to choose from like elu, relu, and tanh. These must match those that are present in the TensorFlow architecture, of course. Feel free to define your own above as well if they are not already included.

    ! Layer X
    if (allocated(layer_output)) deallocate(layer_output)
    allocate(layer_output(layerX_size))
    layer_output = matmul(network(X)%weights, input) + network(X)%biases
    layer_output = activation_function(layer_output)

### Step 4: Modify main.f90

The first thing to change is the definition of the data arrays (e.g., input1(:), input2(:), output_true(:)). Add or remove these depending on how many input parameters you have. If you are not benchmarking this against the TensorFlow model, remove the output_true(:) and other relevant mentions throughout main.f90. Depending on your number of inputs, you will also need to add/remove the variables in charge of the mean and standard deviations used for standardization.

Similarly, the user will specify the actual values for these means and standard deviations that were taken from the output of the TensorFlow Python script. Modify to reflect the number of inputs.

When it comes to the read_dataset calls, you will need to adjust the string to match the names of the datasets in the HDF5 file and the input array you would like to send that information to. The other arguments will be automatically adjusted.

The data_extracted array will then be allocated, with the user setting the integer to the number of inputs + the output. The user will then add the relevant lines to accomodate those input data channels.

If you've picked up on the pattern here, you will need to adjust things to fit the number of inputs that you have. This includes the standardization and unstandardization blocks, X_data and y_data assignments (again no y_data if you are not benchmarking).

### Step 5: Compile

Head back out to the parent directory fortran_dnn_from_tf and compile with the Makefile. Ensure that you have the necessary packages installed: a Fortran compiler (gfortran) and HDF5. These are currently setup as they would be on macOS, but make modifications to the Makefile as needed.

    make

This will produce an executable in the bin/ directory.

### Step 6: Run the Fortran Program

    ```bash
    ./bin/main path/to/datafile path/to/model/parentdir [options]
    ```

Options:
standardize - applicable in most cases, standardizes your incoming data with the specified means/stds
raw_predictions - returns values/statistics on the standardized outputs instead of physical quantities
debug - detailed information to determine where the issue lies

In the case of the example, the number of entries in the testing set is 1000, and the data file that was exported contained physical values, which will need to be standardized.

    ```bash
    ./bin/main example/data/sinusoid_test_data.h5 example/model/sinusoid_model_decomposed standardize
    ```

The example is configured to be a "benchmark" example, which assumes that the outputs are known to compute error metrics for comparison to those generated in the TensorFlow implementation. All of the script components above can be easily altered in the case that you are trying to deploy the model in Fortran to predict novel points without known values. Obviously error metrics would not be available in that case.


## Future Enhancements

	•	Support for Current TensorFlow Model Format: Implement compatibility with the current TensorFlow model save format for broader usability.
	•	Fortran Model Conversion: Develop a Fortran subroutine to replace the Python conversion script, making the workflow more integrated.
	•	Additional Activation Functions: Expand the set of supported activation functions to accommodate more complex models.
	•	User-Specified Network Configuration: Allow users to define network architecture and parameters via an input file, improving flexibility.
	•	OR: automatically takes the network structure/architecture from the model file and configures based on that.
