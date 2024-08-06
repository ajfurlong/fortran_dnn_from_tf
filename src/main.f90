program main
    use read_hdf5_module
    use dnn_module
    use metrics_module
    implicit none

    ! Data arrays
    real(4), allocatable :: input1(:), input2(:), output_true(:)
    real(4), allocatable :: data_extracted(:,:)
    real(4), allocatable :: x_data(:,:), y_data(:), y_pred(:)
    real(4) :: y_temp(1)
    integer :: i, num_entries, num_inputs

    ! Means and standard deviations for standardization
    real(4) :: mean_input1, mean_input2, mean_output_true
    real(4) :: std_input1, std_input2, std_output_true

    ! Timing statistics
    real(4) :: start_time, end_time, elapsed_time

    ! File information
    character(256) :: filename               ! Path to the HDF5 data file
    character(256) :: model_path             ! Path to the parent directory of decomposed layer subdirectories

    ! Options
    logical :: debug = .false.               ! Verbose printing
    logical :: unstandardize_data = .true.   ! If user would like to see raw outputs
    logical :: standardize_data = .false.    ! If taking in physical data, scales to specified means/stds

    ! Command-line arguments
    character(256) :: arg
    integer :: iarg

    ! Check for required arguments
    if (command_argument_count() < 2) then
        print *, 'Error: Filename and model path must be provided.'
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
            case default
                if (trim(arg) == 'raw_predictions') unstandardize_data = .false.
                if (trim(arg) == 'standardize') standardize_data = .true.
                if (trim(arg) == 'debug') debug = .true.
        end select
    end do

    ! Specify the means and standard deviations from tf standardization
    ! These values are currently from the example problem
    mean_input1 = 9.38481456
    mean_input2 = 4.49363963
    mean_output_true = 450.28006616

    std_input1 = 5.43787441
    std_input2 = 0.86546459
    std_output_true = 96.47061761

    call cpu_time(start_time)

    ! Load model weights
    print *, 'Loading model...'
    call load_weights(model_path)
    print *, 'Model load successful.'

    ! Set number of inputs to your architecture's specs
    num_inputs = 2

    ! Read datasets
    print *, 'Reading datasets...'
    call read_dataset(filename, 'input1', input1, num_entries, debug)
    call read_dataset(filename, 'input2', input2, num_entries, debug)
    call read_dataset(filename, 'output_true', output_true, num_entries, debug)

    ! Allocate and combine data into a single array
    allocate(data_extracted(num_entries, num_inputs + 1))
    data_extracted(:, 1) = input1(:)
    data_extracted(:, 2) = input2(:)
    data_extracted(:, 3) = output_true(:)

    ! Standardize datasets if not already in the incoming HDF5 file
    if (standardize_data) then
        print *, 'Standardizing datasets...'
        call standardize(data_extracted(:,1), mean_input1, std_input1)
        call standardize(data_extracted(:,2), mean_input2, std_input2)
        call standardize(data_extracted(:,3), mean_output_true, std_output_true)
    end if

    ! Allocate arrays for non-partitioned data
    allocate(x_data(num_entries, num_inputs))
    allocate(y_data(num_entries))

    ! Assign the data to the respective arrays
    x_data = data_extracted(:, 1:num_inputs)
    y_data = data_extracted(:, num_inputs + 1)

    print *, 'Dataset pre-processing successful.'

    print *, 'Predicting targets...'
    allocate(y_pred(num_entries))

    ! Make and collect predictions
    do i = 1, num_entries
        y_temp = predict(x_data(i, :))
        y_pred(i) = y_temp(1)
    end do

    print *, 'Target prediction successful.'

    ! Unstandardize data to return to physical dimensions (will probably always be needed)
    if (unstandardize_data) then
        call unstandardize(x_data(:,1), mean_input1, std_input1)
        call unstandardize(x_data(:,2), mean_input2, std_input2)
        call unstandardize(y_pred(:), mean_output_true, std_output_true)
        call unstandardize(y_data(:), mean_output_true, std_output_true)
    end if

    ! Timing information
    call cpu_time(end_time)
    elapsed_time = end_time - start_time

    ! Print out metrics if performance-checking with known target y values
    call compute_metrics(y_data, y_pred, elapsed_time)

end program main