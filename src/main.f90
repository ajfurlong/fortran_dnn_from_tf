program main
    use read_hdf5_module
    use dnn_module
    use metrics_module
    implicit none

    ! Parameters
    ! HDF5 length must be specified since hdf5's get_dims inoperable
    integer :: data_length

    ! Data arrays
    ! Add more inputs/outputs depending on your tf architecture
    real(4), allocatable :: input1(:), input2(:), output_true(:)
    real(4), allocatable :: data_extracted(:,:)
    real(4), allocatable :: X_data(:,:), y_data(:), y_pred(:)
    real(4) :: y_temp(1)
    integer :: n

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
    integer :: iarg, i

    ! Check for required arguments
    if (command_argument_count() < 3) then
        print *, 'Error: Filename and data length must be provided.'
        stop
    end if    

    ! Reading command-line arguments
    do iarg = 1, command_argument_count()
        call get_command_argument(iarg, arg)
        select case(iarg)
            case(1)
                filename = trim(arg)
            case(2)
                read(arg, *) data_length
            case(3) 
                model_path = trim(arg)
            case default
                if (trim(arg) == 'raw_predictions') unstandardize_data = .false.
                if (trim(arg) == 'standardize') standardize_data = .true.
                if (trim(arg) == 'debug') debug = .true.
        end select
    end do

    ! Load the means and standard deviations from tf standardization (User specified)
    ! This is typically a characteristic of the model's entire source dataset that is standardized
    ! If you didn't standardize/min-max scale your dataset prior to training, you should for many other reasons!
    ! There is still debate around standardizing outputs in addition to inputs
    ! If you didn't, then remove all relevant lines
    mean_input1 = 9.38497482
    mean_input2 = 4.49366513
    mean_output_true = 450.55616942

    std_input1 = 5.42575189
    std_input2 = 0.86353523
    std_output_true = 96.68878694

    call cpu_time(start_time)

    ! Load model weights
    print *, 'Loading model...'
    call load_weights(model_path)
    print *, 'Model load successful.'

    ! Read datasets
    print *, 'Reading datasets...'
    call read_dataset(filename, 'input1', input1, data_length, n, debug)
    call read_dataset(filename, 'input2', input2, data_length, n, debug)

    ! If true outputs are known (for performance-checking)
    call read_dataset(filename, 'output_true', output_true, data_length, n, debug)

    ! Allocate and combine data into a single array
    allocate(data_extracted(data_length, 3))
    data_extracted(:, 1) = input1(:)
    data_extracted(:, 2) = input2(:)

    ! If true outputs are known (for performance-checking)
    data_extracted(:, 3) = output_true(:)

    ! Standardize datasets if not already
    if (standardize_data) then
        print *, 'Standardizing datasets...'
        call standardize(data_extracted(:,1), mean_input1, std_input1)
        call standardize(data_extracted(:,2), mean_input2, std_input2)

        ! If true outputs are known (for performance-checking)
        call standardize(data_extracted(:,3), mean_output_true, std_output_true)
    end if

    ! Allocate arrays for non-partitioned data
    allocate(X_data(data_length, 2))
    allocate(y_data(data_length))

    ! Assign the data to the respective arrays
    X_data = data_extracted(:, 1:2)
    y_data = data_extracted(:, 3)

    print *, 'Dataset pre-processing successful.'

    print *, 'Predicting targets...'

    allocate(y_pred(data_length))
    
    ! Make and collect predictions
    do i=1,data_length
        y_temp = predict(X_data(i,:))
        y_pred(i) = y_temp(1)
    end do

    print *, 'Target prediction successful.'

    ! If outputs standardized too (look at your tf model)
    if (unstandardize_data) then
        call unstandardize(X_data(:,1), mean_input1, std_input1)
        call unstandardize(X_data(:,2), mean_input2, std_input2)
        call unstandardize(y_pred(:), mean_output_true, std_output_true)

        ! If you are performance-checking with known outputs
        call unstandardize(y_data(:), mean_output_true, std_output_true)
    end if

    ! Timing information
    call cpu_time(end_time)
    elapsed_time = end_time - start_time

    ! Print out metrics if performance-checking
    ! If you are using this to make predictions on unknown data, do not call this
    call compute_metrics(y_data, y_pred, elapsed_time)

end program main