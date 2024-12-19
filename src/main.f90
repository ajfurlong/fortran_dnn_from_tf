program main
    use read_hdf5_module
    use dnn_module
    use metrics_module
    implicit none

    ! Data arrays
    real, allocatable :: input1(:), input2(:)
    real, allocatable :: x_data(:,:), y_pred(:), y_data(:), y_pred_tf(:)
    real :: y_temp(1)
    integer :: i, num_entries
    integer :: num_inputs

    ! Means and standard deviations for standardization
    real, allocatable :: x_mean(:), y_mean(:), x_std(:), y_std(:)

    ! Timing variables
    real :: start_time, end_time, elapsed_time

    ! File information
    character(256) :: filename               ! Path to the HDF5 data file
    character(256) :: model_path             ! Path to the HDF5 model file
    character(256) :: metadata_path          ! Path to the HDF5 metadata file

    ! Options
    logical :: debug = .false.               ! Verbose printing
    logical :: standardize_data = .false.    ! If taking in physical data, scales to specified means/stds

    ! Command-line arguments
    character(256) :: arg
    integer :: iarg

    ! Check for required arguments
    if (command_argument_count() < 2) then
        print *, 'Error: Data, model, and metadata paths must be provided.'
        stop
    end if

    ! Reading command-line arguments
    do iarg = 1, command_argument_count()
        call get_command_argument(iarg, arg)
        select case(iarg)
            case(1)
                filename = trim(arg)
            case(2)
                model_path = trim(arg)
            case(3)
                metadata_path = trim(arg)
            case default
                if (trim(arg) == 'standardize') standardize_data = .true.
                if (trim(arg) == 'debug') debug = .true.
        end select
    end do

    call cpu_time(start_time)

    ! Load model weights, biases and scaling parameters from model.h5 and metadata.h5
    print *, 'Loading model...'
    
    num_inputs = 2
    call initialize_network([2,16,16,1])

    call load_weights(model_path)
    call load_metadata(metadata_path, x_mean, y_mean, x_std, y_std)

    ! If not specified, defaults to relu
    layer_activations(1)%func => relu
    layer_activations(2)%func => relu
    layer_activations(3)%func => no_activation
    print *, 'Model load successful.'

    ! Read datasets from data.h5 file
    print *, 'Reading datasets...'
    call read_dataset(filename, 'input1', input1, num_entries, debug)
    call read_dataset(filename, 'input2', input2, num_entries, debug)
    call read_dataset(filename, 'output_true', y_data, num_entries, debug)
    call read_dataset(filename, 'output_pred', y_pred_tf, num_entries, debug)

    ! Allocate and combine input data into a single array
    allocate(x_data(num_entries, num_inputs))
    x_data(:, 1) = input1(:)
    x_data(:, 2) = input2(:)

    ! Standardize datasets if needed (if you are using physical data)
    if (standardize_data) then
        print *, 'Standardizing datasets...'
        call standardize(x_data(:, 1), x_mean(1), x_std(1))
        call standardize(x_data(:, 2), x_mean(2), x_std(2))
    end if

    print *, 'Dataset pre-processing successful.'

    print *, 'Predicting targets...'
    allocate(y_pred(num_entries))

    ! Make and collect predictions
    do i = 1, num_entries
        y_temp = predict(x_data(i, :))
        y_pred(i) = y_temp(1)
    end do

    print *, 'Target prediction successful.'

    ! Transform output back to physical dimensions
    call unstandardize(y_pred(:), y_mean(1), y_std(1))

    ! Timing information
    call cpu_time(end_time)
    elapsed_time = end_time - start_time

    ! Print out metrics if performance-checking
    call compute_metrics(y_data, y_pred, y_pred_tf, elapsed_time)

end program main