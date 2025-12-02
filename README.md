# Using Pre-Trained TensorFlow DNNs in Fortran

The purpose of this project is to provide a simple yet very accurate framework for using deep neural networks (DNNs) previously trained with TensorFlow to make predictions in a Fortran project. This approach allows Fortran programs to leverage the predictive power of TensorFlow-trained models without requiring complex integration or dependencies beyond HDF5 for data handling. There are a few other projects out there that focus on training and creating DNNs within Fortran, but this project simply does one thing (pretty well) that is easy to understand and implement. The basic idea behind this workflow is to leverage the ease of model training in Python with TensorFlow, with a seamless integration of the trained model into Fortran.

No conversion of any information is required for this script, and it will take model.h5 files directly from the TensorFlow script without outside manipulation.

Relevant verification cases and narrative can be found [here](https://arxiv.org/abs/2502.06853).

## Project Overview

This project offers a minimalistic solution for incorporating TensorFlow-trained DNNs into Fortran applications. The project also includes data processing routines that facilitate benchmarking against original TensorFlow predictions to verify accuracy and consistency. Users are responsible for defining the network architecture and standardization information in a metadata.h5 file on the Python environment-side of things. This is not difficult, but requires adherence to a set format, which is specified below. The system is designed to process large HDF5 files for input and output data.

## Prerequisites

- **Fortran Compiler**: Ensure you have a Fortran compiler installed (e.g., `gfortran`).
- **HDF5 Library**: Required for handling HDF5 files. Install via package manager or from [HDF5 Group](https://www.hdfgroup.org/downloads/hdf5/).
- **Python**: Required for running the model conversion script.
- **TensorFlow**: Install via `pip install tensorflow` for converting models.

## Project Structure

- **bin/**: Directory for storing compiled executables.
- **src/**: Contains Fortran source files, including modules and the main program.
- **obj/**: Stores compiled object and module files.
- **example/**: Example case to train a DNN model using TensorFlow in Python, in this case sinusoid.py.
- **models/**: Stores model weights and biases in a model.h5 file and scaler parameters in a metadata.h5 file.
- **data/**: Stores input and output datasets in a single data.h5 file.
- **output/**: Contains saved "verification mode" performance metrics in a .txt file.

## Usage: Example

The included example, sinusoid.py covers the entire process from training the DNN on the TensorFlow side to running inference within Fortran. The example directory includes the data, model, and source script for a DNN learning a modified sinusoid.

### Step 1: Train the DNN using TensorFlow

The sinusoid.py script should already be configured correctly, so just verify that you have the correct packages installed and run it. This script first generates 5,000 points with a small amount stochastic noise (reproducible with the random seed being set). The data is shuffled, standardized (to mu=0 and sigma=1 to reduce bias from magnitude), and then 80% is used to train the model with 20% being held out for testing. 

The 1,000 test points (inputs, true outputs, and the TensorFlow model's predictions) are exported into an HDF5 file within the example/data/ directory. The model is saved as an HDF5 to example/model/, technically a "legacy" format, but is easy to extract from and will be supported by TensorFlow for the foreseeable future.

In the output of this script, there will be some performance statistics. Directly under these, are the input/output means and standard deviations used during the standardization process. These standardization parameters are automatically stored in the metadata.h5 file saved under models/.

The structure of the metadata.h5 is extremely important but very simple to implement on the Python-side. It consists of three dataset groups: "scaler", "architecture", and "model_info". Each of these names should be pretty self-explanatory. The scaler group contains four datasets which hold information for data standardization: "x_mean", "x_std", "y_mean", and "y_std". The architecture is then specified with a dataset simply listing layer widths ("layer_sizes") and another dataset listing activation functions ("activations", encoded with integers, see below). Model information, such as an ID or string name, are stored in "model_info" under dataset names "model_type_id" and "model_type_name", respectively. Below is a Python snippet that makes this metadata.h5 writing process a whole lot easier:

    with h5py.File(metadata_file, 'w') as f:
        # Scaler information
        grp_scaler = f.create_group("scaler")
        grp_scaler.create_dataset("x_mean", data=x_scaler.mean_)
        grp_scaler.create_dataset("y_mean", data=y_scaler.mean_)
        grp_scaler.create_dataset("x_std", data=x_scaler.scale_)
        grp_scaler.create_dataset("y_std", data=y_scaler.scale_)

        # Architecture information
        grp_arch = f.create_group("architecture")

        # Collect Dense layers only (consistent with Fortran side)
        dense_layers = [layer for layer in model.layers if isinstance(layer, layers.Dense)]

        # Layer sizes: [input_dim, units_layer1, units_layer2, ..., units_last]
        input_dim = model.input_shape[-1]
        layer_sizes = [input_dim] + [layer.units for layer in dense_layers]

        # Activation ID mapping:
        # 0 -> linear (no activation)
        # 1 -> relu
        # 2 -> tanh
        # 3 -> elu
        # 4 -> sigmoid (reserved for future use)
        def _activation_id(layer_obj):
            act = layer_obj.activation
            name = getattr(act, "__name__", str(act))
            if name == "linear":
                return 0
            elif name == "relu":
                return 1
            elif name == "tanh":
                return 2
            elif name == "elu":
                return 3
            elif name == "sigmoid":
                return 4
            else:
                raise ValueError(f"Unsupported activation {name} in layer {layer_obj.name}")

        activation_ids = [_activation_id(layer) for layer in dense_layers]

        grp_arch.create_dataset("layer_sizes", data=np.asarray(layer_sizes, dtype=np.int32))
        grp_arch.create_dataset("activations", data=np.asarray(activation_ids, dtype=np.int32))

        # Model type identifier
        model_info = f.create_group("model_info")
        model_info.create_dataset("model_type_id", data=np.asarray([100], dtype=np.int32))
        model_info.create_dataset("model_type_name", data=np.string_("sinusoid_dnn_test"))

### Step 2: Modify main.f90 network architecture (or whatever module you'd like the predictions to go to)

This is where you will specify the network structure and configuration. This has gotten a lot easier than the first version of the DNN framework, which needed the user to dig through the source network module. The second version was a little easier, but required manual definition of the architecture and activations in main.f90. This version replaces all of that with the built-in capability to read the architecture from the metadata.h5 file. The only thing now left to the user in main.f90 is simply calling load_metadata() and load_weights(). The order DOES matter, since the information contained in metadata.h5 informs load_weights() of the structure.

    call load_metadata(metadata_path, x_mean, y_mean, x_std, y_std)
    call load_weights(model_path)

### Step 3: Modify main.f90

The first thing to change is the definition of the data arrays (i.e., input1(:), input2(:), y_data(:)). Add or remove these depending on how many input parameters you have. If you are not in "verification mode" comparing outputs of the Fortran implementation against those of the TensorFlow model, remove the y_data(:) and y_pred_tf(:) and other relevant mentions throughout main.f90.

When it comes to the read_dataset calls, you will need to adjust the string to match the names of the datasets in the HDF5 file and the input array you would like to send that information to. The other arguments will be automatically adjusted. For our example, where there are two inputs, one true output, and the TensorFlow predicted output, this becomes:

    ! Read datasets from data.h5 file
    print *, 'Reading datasets...'
    call read_dataset(filename, 'input1', input1, num_entries, debug)
    call read_dataset(filename, 'input2', input2, num_entries, debug)
    call read_dataset(filename, 'output_true', y_data, num_entries, debug)
    call read_dataset(filename, 'output_pred', y_pred_tf, num_entries, debug)

The data_extracted array will then be allocated, with the user setting the integer to the number of inputs + the output. The user will then add the relevant lines to accomodate those input data channels:

    ! Allocate and combine input data into a single array
    allocate(x_data(num_entries, num_inputs))
    x_data(:, 1) = input1(:)
    x_data(:, 2) = input2(:)

The list of channels standardized then can be modified as well:

    ! Standardize datasets if needed (if you are using physical data)
    if (standardize_data) then
        print *, 'Standardizing datasets...'
        call standardize(x_data(:, 1), x_mean(1), x_std(1))
        call standardize(x_data(:, 2), x_mean(2), x_std(2))
    end if

All of the necessary modifications should be complete now. Again, this is set up in "verification mode", which assumes that you have known outputs (y_data) and TensorFlow predictions (y_pred_tf), which are then compared against the Fortran predictions (y_pred). If "production mode" is desired, feel free to remove the references of these two variables and compute_metrics(). Using predict() is currently configured to predict a full list of predictions, but you can call it for individual x_data vectors by removing it from that do loop and adjusting the array dimensions elsewhere.

### Step 5: Compile

Head back out to the parent directory fortran_dnn_from_tf and compile with the Makefile. Ensure that you have the necessary packages installed: a Fortran compiler (gfortran) and HDF5. These are currently setup as they would be on macOS, but make modifications to the Makefile as needed.

    make

This will produce an executable in the bin/ directory.

### Step 6: Run the Fortran Program

    ./bin/main path/to/datafile path/to/modelfile path/to/metadatafile [options]

Options:
standardize - applicable in most cases, standardizes your incoming data with the specified means/stds from the metadata file
debug - detailed information to determine where the issue lies

In the case of the example, the number of entries in the testing set is 1000, and the data file that was exported contained physical values, which will need to be standardized.

    ./bin/main data/sinusoid_test_data.h5 models/sinusoid_model_tf.h5 models/sinusoid_metadata_tf.h5 standardize

The example is configured in "verification mode", so it will print a table of performance values and save this table to a .txt file under output/.

## Future Enhancements

	•	Additional Activation Functions: Expand the set of supported activation functions to accommodate more complex models.
	•	Automatic Network Configuration: Allow users to define network architecture and parameters via an input file, improving flexibility.
    •	OR: Configure automatically from the attributes located in the model.h5 file.
	•	Support for Current TensorFlow Model Format: Implement compatibility with the current TensorFlow model save format for broader usability (unlikely due to complexity and lack of necessity).
