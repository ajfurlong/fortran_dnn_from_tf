module dnn_module
    use hdf5
    implicit none
    private
    public :: initialize_network, load_weights, load_metadata, predict, standardize, unstandardize
    public :: relu, tanh_fn, elu, no_activation, layer_activations

    !============================
    ! Abstract interface for activation functions
    !============================
    abstract interface
        function activation_func_interface(x)
            real, dimension(:), intent(in) :: x
            real, dimension(size(x)) :: activation_func_interface
        end function activation_func_interface
    end interface

    ! A type to hold a procedure pointer to an activation function
    type activation_holder
        procedure(activation_func_interface), pointer, nopass :: func
    end type activation_holder

    ! Dynamically sized arrays defining the network structure
    integer, allocatable :: layer_sizes(:)
    ! Each layer has weights and biases
    type layer
        real, allocatable :: weights(:,:)
        real, allocatable :: biases(:)
    end type layer
    type(layer), allocatable :: network(:)

    ! Array of activation holders for each layer
    type(activation_holder), allocatable :: layer_activations(:)

contains

    !------------------------------------------------------------
    ! Initialization routine to define the network architecture
    !------------------------------------------------------------
    subroutine initialize_network(sizes)
        integer, intent(in) :: sizes(:)
        integer :: i, network_depth

        if (allocated(layer_sizes)) deallocate(layer_sizes)
        layer_sizes = sizes
        network_depth = size(layer_sizes) - 1

        if (allocated(network)) deallocate(network)
        allocate(network(network_depth))

        ! Allocate arrays for each layer based on layer_sizes
        do i = 1, network_depth
            allocate(network(i)%weights(layer_sizes(i+1), layer_sizes(i)))
            allocate(network(i)%biases(layer_sizes(i+1)))
        end do

        ! Allocate and assign default activations:
        ! default = relu for all hidden layers, no_activation for the last layer
        if (allocated(layer_activations)) deallocate(layer_activations)
        allocate(layer_activations(network_depth))

        do i = 1, network_depth-1
            layer_activations(i)%func => relu
        end do
        layer_activations(network_depth)%func => no_activation
    end subroutine initialize_network

    !------------------------------------------------------------
    ! Load the network weights from an HDF5 file
    !------------------------------------------------------------
    subroutine load_weights(model_path)
        character(len=*), intent(in) :: model_path
        integer(hid_t) :: file_id
        integer :: i, hdferr
        character(256) :: dataset_name
        integer :: network_depth

        network_depth = size(layer_sizes)-1

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

        ! Load weights and biases for each layer
        do i = 1, network_depth
            ! Kernel weights
            dataset_name = get_dataset_name(i, 'kernel:0')
            call load_dataset(file_id, dataset_name, network(i)%weights)

            ! Biases
            dataset_name = get_dataset_name(i, 'bias:0')
            call load_dataset_1d(file_id, dataset_name, network(i)%biases)
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

    !------------------------------------------------------------
    ! Helper to generate dataset names based on layer index
    ! For a standard Dense layer naming scheme from TF:
    ! The first layer: 'model_weights/dense/dense/kernel:0' and '.../bias:0'
    ! Subsequent layers: 'model_weights/dense_X/dense_X/kernel:0' and '.../bias:0'
    !------------------------------------------------------------
    function get_dataset_name(layer_index, param_name) result(name)
        integer, intent(in) :: layer_index
        character(len=*), intent(in) :: param_name
        character(len=256) :: name

        if (layer_index == 1) then
            write(name, '(A)') 'model_weights/dense/dense/'//trim(param_name)
        else
            write(name, '(A,"/dense_",I0,"/dense_",I0,"/",A)') 'model_weights', layer_index-1, layer_index-1, param_name
        end if
    end function get_dataset_name

    !------------------------------------------------------------
    ! Load metadata for standardization
    !------------------------------------------------------------
    subroutine load_metadata(metadata_path, x_mean, y_mean, x_std, y_std)
        character(len=*), intent(in) :: metadata_path
        real, allocatable, intent(out) :: x_mean(:), y_mean(:), x_std(:), y_std(:)
        integer(hid_t) :: file_id
        integer :: hdferr

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

    !------------------------------------------------------------
    ! Load a 2D dataset from the HDF5 file
    !------------------------------------------------------------
    subroutine load_dataset(file_id, dataset_name, data)
        integer(hid_t), intent(in) :: file_id
        character(len=*), intent(in) :: dataset_name
        real, allocatable, intent(out) :: data(:,:)

        integer(hid_t) :: dataset_id, dataspace_id
        integer :: hdferr
        integer(HSIZE_T) :: dims(2), maxdims(2)

        call h5dopen_f(file_id, dataset_name, dataset_id, hdferr)
        if (hdferr /= 0) then
            print *, 'Error opening dataset:', dataset_name, 'Code:', hdferr
            stop 'Error opening dataset.'
        end if

        call h5dget_space_f(dataset_id, dataspace_id, hdferr)
        if (hdferr /= 0) then
            print *, 'Error getting dataspace for:', dataset_name, 'Code:', hdferr
            stop 'Error getting dataspace.'
        end if

        call h5sget_simple_extent_dims_f(dataspace_id, dims, maxdims, hdferr)
        if (hdferr == -1) then
            print *, 'Error getting dimensions for:', dataset_name
            stop 'Error getting dimensions.'
        end if

        allocate(data(dims(1), dims(2)))
        call h5dread_f(dataset_id, H5T_NATIVE_REAL, data, dims, hdferr)
        if (hdferr /= 0) then
            print *, 'Error reading dataset:', dataset_name, 'Code:', hdferr
            stop 'Error reading dataset.'
        end if

        call h5sclose_f(dataspace_id, hdferr)
        call h5dclose_f(dataset_id, hdferr)
    end subroutine load_dataset

    !------------------------------------------------------------
    ! Load a 1D dataset from the HDF5 file
    !------------------------------------------------------------
    subroutine load_dataset_1d(file_id, dataset_name, data)
        integer(hid_t), intent(in) :: file_id
        character(len=*), intent(in) :: dataset_name
        real, allocatable, intent(out) :: data(:)

        integer(hid_t) :: dataset_id, dataspace_id
        integer :: hdferr
        integer(HSIZE_T) :: dims(1), maxdims(1)

        call h5dopen_f(file_id, dataset_name, dataset_id, hdferr)
        if (hdferr /= 0) then
            print *, 'Error opening dataset:', dataset_name
            stop 'Error opening dataset.'
        end if

        call h5dget_space_f(dataset_id, dataspace_id, hdferr)
        if (hdferr /= 0) then
            print *, 'Error getting dataspace:', dataset_name
            stop 'Error getting dataspace.'
        end if

        call h5sget_simple_extent_dims_f(dataspace_id, dims, maxdims, hdferr)
        if (hdferr == -1) then
            print *, 'Error getting dimensions for:', dataset_name
            stop 'Error getting dimensions.'
        end if

        allocate(data(dims(1)))
        call h5dread_f(dataset_id, H5T_NATIVE_REAL, data, dims, hdferr)
        if (hdferr /= 0) then
            print *, 'Error reading dataset:', dataset_name
            stop 'Error reading dataset.'
        end if

        call h5sclose_f(dataspace_id, hdferr)
        call h5dclose_f(dataset_id, hdferr)
    end subroutine load_dataset_1d

    !------------------------------------------------------------
    ! Standardization and unstandardization routines
    !------------------------------------------------------------
    subroutine standardize(data, mean, std)
        real, intent(inout) :: data(:)
        real, intent(in) :: mean, std
        integer :: i
        do i = 1, size(data)
            data(i) = (data(i) - mean) / std
        end do
    end subroutine standardize

    subroutine unstandardize(data, mean, std)
        real, intent(inout) :: data(:)
        real, intent(in) :: mean, std
        integer :: i
        do i = 1, size(data)
            data(i) = data(i)*std + mean
        end do
    end subroutine unstandardize

    !------------------------------------------------------------
    ! Activation functions
    !------------------------------------------------------------
    function relu(x) result(y)
        real, dimension(:), intent(in) :: x
        real, dimension(size(x)) :: y
        y = max(0.0, x)
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
        where (x < 0.0) y = exp(x) - 1.0
    end function elu

    function no_activation(x) result(y)
        real, dimension(:), intent(in) :: x
        real, dimension(size(x)) :: y
        y = x
    end function no_activation

    !------------------------------------------------------------
    ! Predict using the network:
    ! Each layer uses its own activation function from layer_activations.
    !------------------------------------------------------------
    function predict(input) result(output)
        real, dimension(:), intent(in) :: input
        real, allocatable :: output(:)
        real, allocatable :: current(:), layer_output(:)
        integer :: i, network_depth

        network_depth = size(layer_sizes)-1
        allocate(current(size(input)))
        current = input

        do i = 1, network_depth
            layer_output = matmul(network(i)%weights, current) + network(i)%biases
            layer_output = layer_activations(i)%func(layer_output)
            if (allocated(current)) deallocate(current)
            allocate(current(size(layer_output)))
            current = layer_output
            deallocate(layer_output)
        end do

        allocate(output(size(current)))
        output = current
        deallocate(current)
    end function predict

end module dnn_module