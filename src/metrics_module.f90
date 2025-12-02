module metrics_module
    implicit none
    integer, parameter :: rk = kind(1.0)
    private
    public :: compute_metrics, save_verification_data

contains

    subroutine compute_metrics(y_data, y_pred, y_pred_tf, elapsed_time)
        implicit none
        real(rk), intent(in) :: y_data(:), y_pred(:), y_pred_tf(:), elapsed_time
        real(rk) :: relative_error(size(y_data)), relative_error_tf(size(y_data))
        real(rk) :: abs_err(size(y_data)), abs_err_tf(size(y_data))
        integer :: ferr_above_10, ferr_above_10_tf
        real(rk) :: rrmse, rrmse_tf
        real(rk) :: mae, mape, max_ae, max_ape, min_ae, min_ape
        real(rk) :: mae_tf, mape_tf, max_ae_tf, max_ape_tf, min_ae_tf, min_ape_tf
        real(rk) :: std_ape, ferr_percent, std_ape_tf, ferr_percent_tf
        real(rk) :: ss_total, ss_res, r_squared, ss_res_tf, r_squared_tf
        integer :: unit

        ! Calculate errors and metrics for y_pred
        relative_error = 100.0 * abs((y_data - y_pred) / y_data)
        abs_err = abs(y_pred - y_data)
        ferr_above_10 = count(relative_error > 10.0)
        rrmse = sqrt(sum(((y_pred - y_data) / y_data) ** 2) / size(y_data))

        mae = sum(abs_err) / size(abs_err)
        mape = sum(relative_error) / size(relative_error)
        max_ae = maxval(abs_err)
        max_ape = maxval(relative_error)
        min_ae = minval(abs_err)
        min_ape = minval(relative_error)
        std_ape = sqrt(sum((relative_error - mape)**2) / (size(relative_error) - 1))
        ferr_percent = 100.0 * ferr_above_10 / size(relative_error)

        ! Calculate R^2 for y_pred
        ss_total = sum((y_data - sum(y_data) / size(y_data)) ** 2)
        ss_res = sum((y_data - y_pred) ** 2)
        r_squared = 1.0 - ss_res / ss_total

        ! Calculate errors and metrics for y_pred_tf
        relative_error_tf = 100.0 * abs((y_data - y_pred_tf) / y_data)
        abs_err_tf = abs(y_pred_tf - y_data)
        ferr_above_10_tf = count(relative_error_tf > 10.0)
        rrmse_tf = sqrt(sum(((y_pred_tf - y_data) / y_data) ** 2) / size(y_data))

        mae_tf = sum(abs_err_tf) / size(abs_err_tf)
        mape_tf = sum(relative_error_tf) / size(relative_error_tf)
        max_ae_tf = maxval(abs_err_tf)
        max_ape_tf = maxval(relative_error_tf)
        min_ae_tf = minval(abs_err_tf)
        min_ape_tf = minval(relative_error_tf)
        std_ape_tf = sqrt(sum((relative_error_tf - mape_tf)**2) / (size(relative_error_tf) - 1))
        ferr_percent_tf = 100.0 * ferr_above_10_tf / size(relative_error_tf)

        ! Calculate R^2 for y_pred_tf
        ss_res_tf = sum((y_data - y_pred_tf) ** 2)
        r_squared_tf = 1.0 - ss_res_tf / ss_total

        ! Open a file for writing
        open(unit=unit, file='output/verification_results.txt', status='replace', action='write')
    
        ! Print metrics to both console and file
        print '(A)', "-----------------------------------------------------------------------"
        write(unit, '(A)') "-----------------------------------------------------------------------"
        print '(A, T25, A, T41, A, T66, A)', "Metric", "Fortran", "Benchmark DNN", "Diff"
        write(unit, '(A, T25, A, T41, A, T66, A)') "Metric", "Fortran", "Benchmark DNN", "Diff"
        print '(A)', "-----------------------------------------------------------------------"
        write(unit, '(A)') "-----------------------------------------------------------------------"
        print '(A, T20, F12.6, T40, F12.6, T60, F12.6)', "Mean AE: ", mae, mae_tf, mae - mae_tf
        write(unit, '(A, T20, F12.6, T40, F12.6, T60, F12.6)') "Mean AE: ", mae, mae_tf, mae - mae_tf
        print '(A, T20, F12.6, T40, F12.6, T60, F12.6)', "Max AE: ", max_ae, max_ae_tf, max_ae - max_ae_tf
        write(unit, '(A, T20, F12.6, T40, F12.6, T60, F12.6)') "Max AE: ", max_ae, max_ae_tf, max_ae - max_ae_tf
        print '(A, T20, F12.6, T40, F12.6, T60, F12.6)', "Min AE: ", min_ae, min_ae_tf, min_ae - min_ae_tf
        write(unit, '(A, T20, F12.6, T40, F12.6, T60, F12.6)') "Min AE: ", min_ae, min_ae_tf, min_ae - min_ae_tf
        print '()'
        write(unit, '()')
        print '(A, T20, F12.6, T40, F12.6, T60, F12.6)', "Mean APE (%): ", mape, mape_tf, mape - mape_tf
        write(unit, '(A, T20, F12.6, T40, F12.6, T60, F12.6)') "Mean APE (%): ", mape, mape_tf, mape - mape_tf
        print '(A, T20, F12.6, T40, F12.6, T60, F12.6)', "Max APE (%): ", max_ape, max_ape_tf, max_ape - max_ape_tf
        write(unit, '(A, T20, F12.6, T40, F12.6, T60, F12.6)') "Max APE (%): ", max_ape, max_ape_tf, max_ape - max_ape_tf
        print '(A, T20, F12.6, T40, F12.6, T60, F12.6)', "Min APE (%): ", min_ape, min_ape_tf, min_ape - min_ape_tf
        write(unit, '(A, T20, F12.6, T40, F12.6, T60, F12.6)') "Min APE (%): ", min_ape, min_ape_tf, min_ape - min_ape_tf
        print '(A, T20, F12.6, T40, F12.6, T60, F12.6)', "Std APE (%): ", std_ape, std_ape_tf, std_ape - std_ape_tf
        write(unit, '(A, T20, F12.6, T40, F12.6, T60, F12.6)') "Std APE (%): ", std_ape, std_ape_tf, std_ape - std_ape_tf
        print '()'
        write(unit, '()')
        print '(A, T20, F12.6, T40, F12.6, T60, F12.6)', "rRMSE (%): ", rrmse, rrmse_tf, rrmse - rrmse_tf
        write(unit, '(A, T20, F12.6, T40, F12.6, T60, F12.6)') "rRMSE (%): ", rrmse, rrmse_tf, rrmse - rrmse_tf
        print '(A, T20, F12.6, T40, F12.6, T60, F12.6)', "Ferr > 10% (%): ", ferr_percent, &
                                                        ferr_percent_tf, ferr_percent - ferr_percent_tf
        write(unit, '(A, T20, F12.6, T40, F12.6, T60, F12.6)') "Ferr > 10% (%): ", ferr_percent, &
                                                        ferr_percent_tf, ferr_percent - ferr_percent_tf
        print '(A, T20, I12, T40, I12, T60, I12)', "Ferr > 10% (#): ", ferr_above_10, ferr_above_10_tf, &
                                                        ferr_above_10 - ferr_above_10_tf
        write(unit, '(A, T20, I12, T40, I12, T60, I12)') "Ferr > 10% (#): ", ferr_above_10, ferr_above_10_tf, &
                                                        ferr_above_10 - ferr_above_10_tf
        print '(A, T20, F12.6, T40, F12.6, T60, F12.6)', "R^2: ", r_squared, r_squared_tf, r_squared - r_squared_tf
        write(unit, '(A, T20, F12.6, T40, F12.6, T60, F12.6)') "R^2: ", r_squared, r_squared_tf, r_squared - r_squared_tf
        print '(A)', "-----------------------------------------------------------------------"
        write(unit, '(A)') "-----------------------------------------------------------------------"
        print '(A, T20, I12)', "Total Predictions: ", size(relative_error)
        write(unit, '(A, T20, I12)') "Total Predictions: ", size(relative_error)
        print '(A, T20, F12.6)', "Total CPU Time (s): ", elapsed_time
        write(unit, '(A, T20, F12.6)') "Total CPU Time (s): ", elapsed_time
        print '(A)', "-----------------------------------------------------------------------"
        write(unit, '(A)') "-----------------------------------------------------------------------"
    
        close(unit)
    end subroutine compute_metrics
    
    subroutine save_verification_data(filename, x_data, y_data, y_pred, y_pred_tf, num_entries, num_inputs)
        implicit none
        character(len=*), intent(in) :: filename
        integer, intent(in) :: num_entries, num_inputs
        real(rk), intent(in) :: x_data(num_entries, num_inputs)
        real(rk), intent(in) :: y_data(num_entries)
        real(rk), intent(in) :: y_pred(num_entries)
        real(rk), intent(in) :: y_pred_tf(num_entries)
        
        integer :: i, unit
        character(len=256) :: output_filename
    
        ! Set output file name
        output_filename = trim(filename)
    
        ! Open file for writing
        open(newunit=unit, file=output_filename, status="replace", action="write")
    
        ! Write header
        write(unit, '(A)') "input1,input2,true_output,fortran_pred,tensorflow_pred"
    
        ! Write data entries
        do i = 1, num_entries
            write(unit, '(F10.6, ",", F10.6, ",", F14.10, ",", F14.10, ",", F14.10)') &
                x_data(i, 1), x_data(i, 2), y_data(i), y_pred(i), y_pred_tf(i)
        end do
    
        ! Close file
        close(unit)
        print *, "Verification data saved to ", trim(output_filename)
    end subroutine save_verification_data

end module metrics_module