# Changelog

## [1.3.0] - 20224-08-08
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

## [1.2.0] - 2024-08-06
### Added
- Automatic dataset length detection in `read_hdf5_module`, requiring a "num_entries" attribute on each dataset in the HDF5 data file.
- Print total number of predictions made in `metrics_module`.
- Parameter "num_inputs" in `main` to automate data array allocation post-definition.

### Changed
- Cleaned up structure and comments in `main`.

### Fixed
- Corrected how the model file input argument is processed in `dnn_module`, which required a trailing "/".

## [1.1.0] - 2024-08-05
### Added
- Initial release.