module dnn_module
    implicit none
    private
    public :: load_weights, predict, standardize, unstandardize

    ! Basic network information
    ! Must be exactly the same as your source model from tf
    ! Number of hidden layers + output layer
    integer, parameter :: network_depth = 3

    ! Define the sizes of each layer (number of neurons)
    integer, parameter :: input_size = 2
    integer, parameter :: layer1_size = 16
    integer, parameter :: layer2_size = 16
    integer, parameter :: output_size = 1

    ! Define arrays for weights and biases
    type layer
        real(4), allocatable :: weights(:,:)
        real(4), allocatable :: biases(:)
    end type layer
    type(layer), dimension(network_depth) :: network

contains

    subroutine load_weights(model_path)
        integer :: i, ios
        character(256), intent(in) :: model_path
        character(256) :: weight_filename, bias_filename

        ! Allocate memory for weights and biases for each layer
        ! This is the only section to alter in this subroutine
        allocate(network(1)%weights(layer1_size, input_size))
        allocate(network(1)%biases(layer1_size))

        allocate(network(2)%weights(layer2_size, layer1_size))
        allocate(network(2)%biases(layer2_size))

        allocate(network(3)%weights(output_size, layer2_size))
        allocate(network(3)%biases(output_size))

        ! Automatically load weights and biases for each layer
        do i = 1, network_depth
            if (i == 1) then
                weight_filename = trim(model_path) // 'dense/dense/kernel:0.txt'
                bias_filename = trim(model_path) // 'dense/dense/bias:0.txt'
                print *, model_path
            else
                write(weight_filename, '(A, "dense_", I0, "/", "dense_", I0, "/kernel:0.txt")') &
                                                                    trim(model_path), i-1, i-1
                write(bias_filename, '(A, "dense_", I0, "/", "dense_", I0, "/bias:0.txt")') &
                                                                    trim(model_path), i-1, i-1

            end if

            open(unit=10, file=weight_filename, status='old', action='read', iostat=ios)
            if (ios /= 0) stop 'Error opening weight file'
            read(10, *) network(i)%weights
            close(10)

            open(unit=11, file=bias_filename, status='old', action='read', iostat=ios)
            if (ios /= 0) stop 'Error opening bias file'
            read(11, *) network(i)%biases
            close(11)
        end do
    end subroutine load_weights

    ! Standardizes to the declared means and stds of original training set, typically inputs and outputs
    subroutine standardize(data, mean, std)
        real(4), intent(inout) :: data(:)
        real(4), intent(in) :: mean, std
        integer :: i

        do i = 1, size(data)
            data(i) = (data(i) - mean) / std
        end do
    end subroutine standardize
    
    ! Unstandardizes, used after predictions are made to return values to physical dimensions
    subroutine unstandardize(data, mean, std)
        real(4), intent(inout) :: data(:)
        real(4), intent(in) :: mean, std
        integer :: i
    
        do i = 1, size(data)
            data(i) = data(i) * std + mean
        end do
    end subroutine unstandardize

    ! Possible activation functions (feel free to add more!)
    function relu(x) result(y)
        real(4), dimension(:), intent(in) :: x
        real(4), dimension(size(x)) :: y
        y = max(0.0d0, x)
    end function relu

    function tanh_fn(x) result(y)
        real(4), dimension(:), intent(in) :: x
        real(4), dimension(size(x)) :: y
        y = tanh(x)
    end function tanh_fn

    function elu(x) result(y)
        real(4), dimension(:), intent(in) :: x
        real(4), dimension(size(x)) :: y
        y = x
        where (x < 0.0d0) y = exp(x) - 1.0d0
    end function elu

    function predict(input) result(output)
        real(4), dimension(input_size), intent(in) :: input
        real(4), dimension(output_size) :: output
        real(4), dimension(:), allocatable :: layer_output
        real(4), dimension(:), allocatable :: temp_output

        ! Layer 1
        if (allocated(layer_output)) deallocate(layer_output)
        allocate(layer_output(layer1_size))
        layer_output = matmul(network(1)%weights, input) + network(1)%biases
        layer_output = relu(layer_output)
    
        ! Layer 2
        if (allocated(temp_output)) deallocate(temp_output)
        allocate(temp_output(layer2_size))
        temp_output = matmul(network(2)%weights, layer_output) + network(2)%biases
        layer_output = relu(temp_output)
        deallocate(temp_output)
    
        ! Output Layer
        if (allocated(temp_output)) deallocate(temp_output)
        allocate(temp_output(output_size))
        temp_output = matmul(network(3)%weights, layer_output) + network(3)%biases
        output = temp_output
        deallocate(temp_output)
    
        deallocate(layer_output)
    end function predict

end module dnn_module