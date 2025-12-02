module read_hdf5_module
    use hdf5
    use iso_fortran_env, only: r4 => real32, r8 => real64
    implicit none
    integer, parameter :: rk = kind(1.0)
    private
    public :: read_dataset
    
contains

    subroutine read_dataset(filename, dataset_name, data, num_entries, debug)

        character(len=*), intent(in) :: filename, dataset_name
        real(rk), allocatable, intent(out) :: data(:)
        integer, intent(out) :: num_entries
        logical, intent(in) :: debug

        integer(hid_t) :: file_id, dataset_id, dataspace_id, type_id
        integer :: hdferr
        integer(HSIZE_T), dimension(1) :: ddims, maxdims
        integer(HSIZE_T) :: type_size
        integer :: class_id

        real(r4), allocatable :: buf_r4(:)   ! Buffer for single precision
        real(r8), allocatable :: buf_r8(:)   ! Buffer for double precision

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

        ! Automatically detect precision & read accordingly using explicit buffers,
        ! then convert to the module's internal kind rk.
        select case (type_size)
        case (4_HSIZE_T)  ! Dataset stored in REAL(4)
            allocate(buf_r4(num_entries))
            call h5dread_f(dataset_id, H5T_NATIVE_REAL, buf_r4, ddims, hdferr)
            data = real(buf_r4, kind=rk)
            deallocate(buf_r4)

        case (8_HSIZE_T)  ! Dataset stored in REAL(8)
            allocate(buf_r8(num_entries))
            call h5dread_f(dataset_id, H5T_NATIVE_DOUBLE, buf_r8, ddims, hdferr)
            data = real(buf_r8, kind=rk)
            deallocate(buf_r8)

        case default
            print *, "Error: Unsupported dataset type size:", type_size
            stop
        end select

        ! Cleanup
        call h5tclose_f(type_id, hdferr)
        call h5sclose_f(dataspace_id, hdferr)
        call h5dclose_f(dataset_id, hdferr)
        call h5fclose_f(file_id, hdferr)
        call h5close_f(hdferr)

    end subroutine read_dataset

end module read_hdf5_module