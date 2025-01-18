module read_hdf5_module
    use hdf5
    use iso_fortran_env, only: r4 => real32, r8 => real64
    implicit none
    private
    public :: read_dataset
    
contains

    subroutine read_dataset(filename, dataset_name, data, num_entries, debug)

        character(len=*), intent(in) :: filename, dataset_name
        real, allocatable, intent(out) :: data(:)  
        integer, intent(out) :: num_entries
        logical, intent(in) :: debug

        integer(hid_t) :: file_id, dataset_id, dataspace_id, type_id
        integer :: hdferr
        integer(HSIZE_T), dimension(1) :: ddims, maxdims
        integer(HSIZE_T) :: type_size
        integer :: class_id
        integer, parameter :: default_precision = kind(1.0)

        real(r4), allocatable :: data_single(:)  
        real(r8), allocatable :: data_double(:)  

        ! Open HDF5 file and dataset
        call h5open_f(hdferr)
        call h5fopen_f(filename, H5F_ACC_RDONLY_F, file_id, hdferr)
        call h5dopen_f(file_id, dataset_name, dataset_id, hdferr)
        call h5dget_space_f(dataset_id, dataspace_id, hdferr)

        ! Get dataset dimensions
        call h5sget_simple_extent_dims_f(dataspace_id, ddims, maxdims, hdferr)
        num_entries = ddims(1)

        ! Get dataset type
        call h5dget_type_f(dataset_id, type_id, hdferr)
        call h5tget_size_f(type_id, type_size, hdferr)
        call h5tget_class_f(type_id, class_id, hdferr)

        if (class_id /= H5T_FLOAT_F) then
            print *, "Error: Dataset is NOT floating-point!"
            stop
        end if

        ! Allocate output array
        allocate(data(num_entries))

        ! Automatically detect precision & read accordingly
        select case (type_size)
            case (4)  ! Dataset stored in REAL(4)
                if (default_precision == 4) then
                    call h5dread_f(dataset_id, H5T_NATIVE_REAL, data, ddims, hdferr)
                else
                    allocate(data_single(num_entries))
                    call h5dread_f(dataset_id, H5T_NATIVE_REAL, data_single, ddims, hdferr)
                    data = real(data_single, kind=kind(data(1)))
                    deallocate(data_single)
                end if

            case (8)  ! Dataset stored in REAL(8)
                if (default_precision == 8) then
                    call h5dread_f(dataset_id, H5T_NATIVE_DOUBLE, data, ddims, hdferr)
                else
                    allocate(data_double(num_entries))
                    call h5dread_f(dataset_id, H5T_NATIVE_DOUBLE, data_double, ddims, hdferr)
                    data = real(data_double, kind=kind(data(1)))
                    deallocate(data_double)
                end if

            case default
                print *, "Error: Unsupported dataset type size:", type_size
                stop
        end select

        ! Cleanup
        call h5tclose_f(type_id, hdferr)
        call h5sclose_f(dataspace_id, hdferr)
        call h5dclose_f(dataset_id, hdferr)
        call h5fclose_f(file_id, hdferr)

    end subroutine read_dataset

end module read_hdf5_module