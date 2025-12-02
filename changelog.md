# Changelog

## [1.5.0] - 2025-12-02
### Added
- The ability to load model architectures automatically using information now in the metadata.h5 format rules
- User does not need to define architectures manually anymore

### Changed
- Simplified real precision handling throughout dnn_module.f90 and read_hdf5_module.f90
- Only load_metadata() and load_weights() are strictly needed in terms of model architecture definitions
- A specified format is REQUIRED for the metadata.h5 files, which allows for autoloading to work

### Fixed
- Strengthened the brittle precision logic

## [1.4.0] - 2025-01-18
### Added
- Verification toy problem "nonlinear regression" and results
- New subroutine save_verification_data() added to metrics_module.f90 to allow for data analysis outside of Fortran

### Changed
- HDF5 processing updated to detect if file dataset is F32 or F64 (TensorFlow model.save() defaults to F32)
- Casts HDF5 read-in datasets to correct precision if different than the compiled
- Activation function names updated "relu", "elu" -> "relu_fn", "elu_fn"

### Fixed
- Corrected support for double precision

## [1.3.0] - 2024-12-19
### Added
- Ability to structure the network architecture entirely from inside main.f90
- All layer and activation-related components now automatically build and allocate in dnn_module
- Activation functions now set with pointers for each layer in main.f90

### Changed
- Major refactor of the dnn_module for efficiency and readability
- Removed any hard-coding within dnn_module for layer and activation structuring
- General loading and naming routines updated, such as get_dataset_name

### Fixed
- N/A

## [1.2.0] - 2024-08-08
### Added
- Support for directly using the model.h5 file in load_weights() instead of having to convert to text file beforehand.
- Standardization parameters now load directly in from a metadata.h5 file instead of manual specification in source.
- Saves "verification mode" results in output/verification_results.txt.

### Changed
- Modified the predict() function structure in dnn_module to avoid successive allocation and deallocation of the temp_output and layer_output arrays.
- Removed unnecessary calls of standardize and unstandardize on verification output channels in main.f90.
- Removed the option to NOT unstandardize the data (extremely rare case that it would ever be used).
- Uses native HDF5 call to find verification dataset dimensions instead of needing a num_entries attribute in the data.h5 file.
- Changed input arguments to allow for the specification of model.h5 and metadata.h5 files instead of text directories.
- Eliminated extraneous variables and improved simplicity of data handling.

### Fixed
- Removed all real(4) single-type specification to allow for compiler-specified precision.

## [1.1.0] - 2024-08-06
### Added
- Automatic dataset length detection in `read_hdf5_module`, requiring a "num_entries" attribute on each dataset in the HDF5 data file.
- Print total number of predictions made in `metrics_module`.
- Parameter "num_inputs" in `main` to automate data array allocation post-definition.

### Changed
- Cleaned up structure and comments in `main`.

### Fixed
- Corrected how the model file input argument is processed in `dnn_module`, which required a trailing "/".

## [1.0.0] - 2024-08-05
### Added
- Initial release.