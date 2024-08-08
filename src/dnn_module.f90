module dnn_module
    implicit none
    private
    public :: load_weights, load_metadata, predict, standardize, unstandardize

    ! Basic network information
    ! Must be exactly the same as your source model from tf
    ! Number of hidden layers + output layer
    integer, parameter :: network_depth = 3

    ! Define the sizes of each layer (number of neurons)
    integer, parameter :: input_size = 2
    integer, parameter :: layer1_size = 16
    integer, parameter :: layer2_size = 16
    integer, parameter :: output_size = 1
    integer, parameter :: max_layer_size = max(input_size, layer1_size, layer2_size, output_size)

    ! Define arrays for weights and biases
    type layer
        real, allocatable :: weights(:,:)
        real, allocatable :: biases(:)
    end type layer
    type(layer), dimension(network_depth) :: network

contains

    subroutine load_weights(model_path)
        use hdf5
        implicit none
        integer :: i, hdferr
        character(256), intent(in) :: model_path
        character(256) :: dataset_name
        integer(hid_t) :: file_id

        ! Open the HDF5 file
        call h5open_f(hdferr)
        if (hdferr /= 0) then
            print *, 'Error opening HDF5 library:', hdferr
            stop 'Error opening HDF5 library.'
        end if

        call h5fopen_f(trim(model_path), H5F_ACC_RDONLY_F, file_id, hdferr)
        if (hdferr /= 0) then
            print *, 'Error opening HDF5 file:', model_path, 'Error code:', hdferr
            stop 'Error opening HDF5 file.'
        end if

        ! Allocate memory for weights and biases for each layer
        allocate(network(1)%weights(layer1_size, input_size))
        allocate(network(1)%biases(layer1_size))

        allocate(network(2)%weights(layer2_size, layer1_size))
        allocate(network(2)%biases(layer2_size))

        allocate(network(3)%weights(output_size, layer2_size))
        allocate(network(3)%biases(output_size))

        ! Automatically load weights and biases for each layer
        do i = 1, network_depth
            if (i == 1) then
                dataset_name = 'model_weights/dense/dense/kernel:0'
                call load_dataset(file_id, dataset_name, network(i)%weights)
                dataset_name = 'model_weights/dense/dense/bias:0'
                call load_dataset_1d(file_id, dataset_name, network(i)%biases)
            else
                write(dataset_name, '(A, "/dense_", I0, "/dense_", I0, "/kernel:0")') 'model_weights', i-1, i-1
                call load_dataset(file_id, dataset_name, network(i)%weights)
    
                write(dataset_name, '(A, "/dense_", I0, "/dense_", I0, "/bias:0")') 'model_weights', i-1, i-1
                call load_dataset_1d(file_id, dataset_name, network(i)%biases)
            end if
        end do

        ! Close the HDF5 file
        call h5fclose_f(file_id, hdferr)
        if (hdferr /= 0) then
            print *, 'Error closing HDF5 file:', model_path, 'Error code:', hdferr
            stop 'Error closing HDF5 file.'
        end if

        call h5close_f(hdferr)
        if (hdferr /= 0) then
            print *, 'Error closing HDF5 library:', hdferr
            stop 'Error closing HDF5 library.'
        end if
    end subroutine load_weights

    subroutine load_metadata(metadata_path, x_mean, y_mean, x_std, y_std)
        use hdf5
        implicit none
        character(len=*), intent(in) :: metadata_path
        real, allocatable, intent(out) :: x_mean(:), y_mean(:), x_std(:), y_std(:)
        integer :: hdferr
        integer(hid_t) :: file_id
    
        ! Open the HDF5 metadata file
        call h5open_f(hdferr)
        if (hdferr /= 0) then
            print *, 'Error opening HDF5 library: ', hdferr
            stop 'Error opening HDF5 library.'
        end if
    
        call h5fopen_f(metadata_path, H5F_ACC_RDONLY_F, file_id, hdferr)
        if (hdferr /= 0) then
            print *, 'Error opening HDF5 file:', metadata_path, 'Error code:', hdferr
            stop 'Error opening HDF5 file.'
        end if
    
        ! Read scaler information
        call load_dataset_1d(file_id, 'scaler/x_mean', x_mean)
        call load_dataset_1d(file_id, 'scaler/y_mean', y_mean)
        call load_dataset_1d(file_id, 'scaler/x_std', x_std)
        call load_dataset_1d(file_id, 'scaler/y_std', y_std)
    
        ! Close the HDF5 file
        call h5fclose_f(file_id, hdferr)
        if (hdferr /= 0) then
            print *, 'Error closing HDF5 file:', metadata_path, 'Error code:', hdferr
            stop 'Error closing HDF5 file.'
        end if
    
        call h5close_f(hdferr)
        if (hdferr /= 0) then
            print *, 'Error closing HDF5 library: ', hdferr
            stop 'Error closing HDF5 library.'
        end if
    end subroutine load_metadata

    subroutine load_dataset(file_id, dataset_name, data)
        use hdf5
        implicit none
        integer(hid_t), intent(in) :: file_id
        character(len=*), intent(in) :: dataset_name
        real, allocatable, intent(out) :: data(:,:)
        integer(hid_t) :: dataset_id, dataspace_id
        integer :: hdferr
        integer(HSIZE_T) :: dims(2), maxdims(2)  ! Correct type for HDF5 dimensions
        integer :: rank
    
        ! Open the dataset
        call h5dopen_f(file_id, dataset_name, dataset_id, hdferr)
        if (hdferr /= 0) then
            print *, 'Error opening dataset:', dataset_name, 'Error code:', hdferr
            stop 'Error opening dataset.'
        end if
    
        ! Get the dataspace
        call h5dget_space_f(dataset_id, dataspace_id, hdferr)
        if (hdferr /= 0) then
            print *, 'Error getting dataspace for dataset:', dataset_name, 'Error code:', hdferr
            stop 'Error getting dataspace.'
        end if
    
        ! Get the dimensions of the dataset
        call h5sget_simple_extent_dims_f(dataspace_id, dims, maxdims, hdferr)
        if (hdferr == -1) then
            print *, 'Error getting dimensions for dataset:', dataset_name, 'Error code:', hdferr
            stop 'Error getting dimensions.'
        end if
    
        ! Allocate the array to read the data
        allocate(data(dims(1), dims(2)))
    
        ! Read the data from the dataset
        call h5dread_f(dataset_id, H5T_NATIVE_REAL, data, dims, hdferr)
        if (hdferr /= 0) then
            print *, 'Error reading dataset:', dataset_name, 'Error code:', hdferr
            stop 'Error reading dataset.'
        end if
    
        ! Close the dataspace and dataset
        call h5sclose_f(dataspace_id, hdferr)
        if (hdferr /= 0) then
            print *, 'Error closing dataspace for dataset:', dataset_name, 'Error code:', hdferr
            stop 'Error closing dataspace.'
        end if
    
        call h5dclose_f(dataset_id, hdferr)
        if (hdferr /= 0) then
            print *, 'Error closing dataset:', dataset_name, 'Error code:', hdferr
            stop 'Error closing dataset.'
        end if
    end subroutine load_dataset
    
    subroutine load_dataset_1d(file_id, dataset_name, data)
        use hdf5
        implicit none
        integer(hid_t), intent(in) :: file_id
        character(len=*), intent(in) :: dataset_name
        real, allocatable, intent(out) :: data(:)
        integer(hid_t) :: dataset_id, dataspace_id
        integer :: hdferr
        integer(HSIZE_T) :: dims(1), maxdims(1)  ! Correct type for HDF5 dimensions
        integer :: rank
    
        ! Open the dataset
        call h5dopen_f(file_id, dataset_name, dataset_id, hdferr)
        if (hdferr /= 0) then
            print *, 'Error opening dataset:', dataset_name, 'Error code:', hdferr
            stop 'Error opening dataset.'
        end if
    
        ! Get the dataspace
        call h5dget_space_f(dataset_id, dataspace_id, hdferr)
        if (hdferr /= 0) then
            print *, 'Error getting dataspace for dataset:', dataset_name, 'Error code:', hdferr
            stop 'Error getting dataspace.'
        end if
    
        ! Get the dimensions of the dataset
        call h5sget_simple_extent_dims_f(dataspace_id, dims, maxdims, hdferr)
        if (hdferr == -1) then
            print *, 'Error getting dimensions for dataset:', dataset_name, 'Error code:', hdferr
            stop 'Error getting dimensions.'
        end if
    
        ! Allocate the array to read the data
        allocate(data(dims(1)))
    
        ! Read the data from the dataset
        call h5dread_f(dataset_id, H5T_NATIVE_REAL, data, dims, hdferr)
        if (hdferr /= 0) then
            print *, 'Error reading dataset:', dataset_name, 'Error code:', hdferr
            stop 'Error reading dataset.'
        end if
    
        ! Close the dataspace and dataset
        call h5sclose_f(dataspace_id, hdferr)
        if (hdferr /= 0) then
            print *, 'Error closing dataspace for dataset:', dataset_name, 'Error code:', hdferr
            stop 'Error closing dataspace.'
        end if
    
        call h5dclose_f(dataset_id, hdferr)
        if (hdferr /= 0) then
            print *, 'Error closing dataset:', dataset_name, 'Error code:', hdferr
            stop 'Error closing dataset.'
        end if
    end subroutine load_dataset_1d

    function itoa(i)
        integer, intent(in) :: i
        character(len=20) :: itoa
        write(itoa, '(I0)') i
    end function itoa

    ! Standardizes to the declared means and stds of original training set, typically inputs and outputs
    subroutine standardize(data, mean, std)
        real, intent(inout) :: data(:)
        real, intent(in) :: mean, std
        integer :: i

        do i = 1, size(data)
            data(i) = (data(i) - mean) / std
        end do
    end subroutine standardize
    
    ! Unstandardizes, used after predictions are made to return values to physical dimensions
    subroutine unstandardize(data, mean, std)
        real, intent(inout) :: data(:)
        real, intent(in) :: mean, std
        integer :: i
    
        do i = 1, size(data)
            data(i) = data(i) * std + mean
        end do
    end subroutine unstandardize

    ! Possible activation functions (feel free to add more!)
    function relu(x) result(y)
        real, dimension(:), intent(in) :: x
        real, dimension(size(x)) :: y
        y = max(0.0d0, x)
    end function relu

    function tanh_fn(x) result(y)
        real, dimension(:), intent(in) :: x
        real, dimension(size(x)) :: y
        y = tanh(x)
    end function tanh_fn

    function elu(x) result(y)
        real, dimension(:), intent(in) :: x
        real, dimension(size(x)) :: y
        y = x
        where (x < 0.0d0) y = exp(x) - 1.0d0
    end function elu

    function predict(input) result(output)
        real, dimension(input_size), intent(in) :: input
        real, dimension(output_size) :: output
        real, dimension(max_layer_size) :: layer_output, temp_output
    
        ! Layer 1
        layer_output(1:layer1_size) = matmul(network(1)%weights, input) + network(1)%biases
        layer_output(1:layer1_size) = relu(layer_output(1:layer1_size))
    
        ! Layer 2
        temp_output(1:layer2_size) = matmul(network(2)%weights, layer_output(1:layer1_size)) + network(2)%biases
        layer_output(1:layer2_size) = relu(temp_output(1:layer2_size))
    
        ! Output Layer
        temp_output(1:output_size) = matmul(network(3)%weights, layer_output(1:layer2_size)) + network(3)%biases
        output = temp_output(1:output_size)

    end function predict

end module dnn_module