# Changelog

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