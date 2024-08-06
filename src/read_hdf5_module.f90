module read_hdf5_module
    use hdf5
    implicit none
    private
    public :: read_dataset
    
contains

    subroutine read_dataset(filename, dataset_name, data, num_entries, debug)
        character(len=*), intent(in) :: filename, dataset_name
        real(4), allocatable, intent(out) :: data(:)
        integer, intent(inout) :: num_entries
        integer(hid_t) :: file_id, dataset_id, dataspace_id, attr_id
        integer :: rank, hdferr
        logical, intent(in) :: debug
        logical :: space_type

        ! Define constants explicitly
        integer, parameter :: H5F_ACC_RDONLY_F = 0

        ! Manually define dimensions
        integer(HSIZE_T), dimension(1) :: ddims, adims

        ! Open the HDF5 file and read the datasets
        call h5open_f(hdferr)
        if (hdferr /= 0) then
            print *, 'Error opening HDF5 library: ', hdferr
            stop 'Error opening HDF5 library.'
        end if

        ! Open the HDF5 file
        call h5fopen_f(filename, H5F_ACC_RDONLY_F, file_id, hdferr)
        if (hdferr /= 0) then
            print *, 'Error opening HDF5 file:', filename, 'Error code:', hdferr
            stop 'Error opening HDF5 file.'
        end if
        if (debug) print *, 'Successfully opened HDF5 file: ', filename

        ! Open the dataset
        call h5dopen_f(file_id, dataset_name, dataset_id, hdferr)
        if (hdferr /= 0) then
            print *, 'Error opening dataset:', dataset_name, 'Error code:', hdferr
            stop 'Error opening dataset.'
        end if
        if (debug) print *, 'Successfully opened dataset: ', dataset_name

        ! Get the dataspace
        call h5dget_space_f(dataset_id, dataspace_id, hdferr)
        if (hdferr /= 0) then
            print *, 'Error getting dataspace for dataset: ', dataset_name, 'Error code: ', hdferr
            stop 'Error getting dataspace.'
        end if
        if (debug) print *, 'Successfully got dataspace for dataset: ', dataset_name

        ! Check the dataspace type
        call h5sis_simple_f(dataspace_id, space_type, hdferr)
        if (hdferr /= 0) then
            print *, 'Error determining if dataspace is simple: ', 'Error code: ', hdferr
            stop 'Error determining if dataspace is simple.'
        else if (.not. space_type) then
            print *, 'Dataspace is not simple for dataset: ', dataset_name
            stop 'Dataspace is not simple.'
        end if
        if (debug) print *, 'Dataspace is simple for dataset: ', dataset_name

        ! Open and read the num_entries attribute
        call h5aopen_name_f(dataset_id, 'num_entries', attr_id, hdferr)
        if (hdferr /= 0) then
            print *, 'Error opening attribute num_entries for dataset:', dataset_name
            call h5dclose_f(dataset_id, hdferr)
            call h5fclose_f(file_id, hdferr)
            stop 'Attribute open error'
        end if

        adims = 1

        call h5aread_f(attr_id, H5T_NATIVE_INTEGER, num_entries, adims, hdferr)
        if (hdferr /= 0) then
            print *, 'Error reading attribute num_entries for dataset:', dataset_name
            call h5aclose_f(attr_id, hdferr)
            call h5dclose_f(dataset_id, hdferr)
            call h5fclose_f(file_id, hdferr)
            stop 'Attribute read error'
        end if

        call h5aclose_f(attr_id, hdferr)

        ! Allocate the array to read the data
        ddims = num_entries
        allocate(data(ddims(1)))

        ! Read the data from the dataset
        call h5dread_f(dataset_id, H5T_NATIVE_REAL, data, ddims, hdferr)
        if (hdferr /= 0) then
            print *, 'Error reading dataset: ', dataset_name, 'Error code: ', hdferr
            call h5sclose_f(dataspace_id, hdferr)
            call h5dclose_f(dataset_id, hdferr)
            call h5fclose_f(file_id, hdferr)
            stop 'Error reading dataset.'
        end if
        print *, 'Successfully read dataset: ', dataset_name

        ! Close the dataspace
        call h5sclose_f(dataspace_id, hdferr)
        if (hdferr /= 0) then
            print *, 'Error closing dataspace for dataset: ', dataset_name, 'Error code: ', hdferr
            call h5dclose_f(dataset_id, hdferr)
            call h5fclose_f(file_id, hdferr)
            stop 'Error closing dataspace.'
        end if
        if (debug) print *, 'Successfully closed dataspace for dataset: ', dataset_name

        ! Close the dataset
        call h5dclose_f(dataset_id, hdferr)
        if (hdferr /= 0) then
            print *, 'Error closing dataset: ', dataset_name, 'Error code: ', hdferr
            call h5fclose_f(file_id, hdferr)
            stop 'Error closing dataset.'
        end if
        if (debug) print *, 'Successfully closed dataset: ', dataset_name
        
        ! Close the file
        call h5fclose_f(file_id, hdferr)
        if (hdferr /= 0) then
            print *, 'Error closing HDF5 file: ', filename, 'Error code: ', hdferr
            stop 'Error closing HDF5 file.'
        end if
        if (debug) print *, 'Successfully closed HDF5 file: ', filename
    end subroutine read_dataset

end module read_hdf5_module