module metrics_module
    implicit none
    private
    public :: compute_metrics

contains

subroutine compute_metrics(y_data, y_pred, elapsed_time)
    implicit none
    real(4), intent(in) :: y_data(:), y_pred(:), elapsed_time
    real(4) :: relative_error(size(y_data))
    real(4) :: abs_err(size(y_data))
    integer :: ferr_above_10
    real(4) :: rrmse
    real(4) :: mae, mape, max_ae, max_ape, min_ae, min_ape
    real(4) :: std_ape, ferr_percent
    real(4) :: ss_total, ss_res, r_squared

    ! Calculate errors and metrics for pred_CHF
    relative_error = 100.0 * abs((y_data - y_pred) / y_data)
    abs_err = abs(y_pred - y_data)
    ferr_above_10 = count(relative_error > 10.0)
    rrmse = 100 * sqrt(sum(((y_pred - y_data) / y_data) ** 2) / size(y_data))

    mae = sum(abs_err) / size(abs_err)
    mape = sum(relative_error) / size(relative_error)
    max_ae = maxval(abs_err)
    max_ape = maxval(relative_error)
    min_ae = minval(abs_err)
    min_ape = minval(relative_error)
    std_ape = sqrt(sum((relative_error - mape)**2) / (size(relative_error) - 1))
    ferr_percent = 100.0 * ferr_above_10 / size(relative_error)

    ! Calculate R^2 for pred_CHF
    ss_total = sum((y_data - sum(y_data) / size(y_data)) ** 2)
    ss_res = sum((y_data - y_pred) ** 2)
    r_squared = 1.0 - ss_res / ss_total

    print '(A)', "-----------------------------------------------------------------------"
    print '(A, T25, A)', "Metric", "Fortran"
    print '(A)', "-----------------------------------------------------------------------"
    ! print '(A, T20, F12.6)', "Mean AE: ", mae
    ! print '(A, T20, F12.6)', "Max AE: ", max_ae
    ! print '(A, T20, F12.6)', "Min AE: ", min_ae
    print '(A, T20, F12.6)', "Mean APE (%): ", mape
    print '(A, T20, F12.6)', "Max APE (%): ", max_ape
    print '(A, T20, F12.6)', "Min APE (%): ", min_ape
    print '(A, T20, F12.6)', "Std APE (%): ", std_ape
    print '(A, T20, F12.6)', "rRMSE (%): ", rrmse
    print '(A, T20, F12.6)', "Ferr > 10% (%): ", ferr_percent
    ! print '(A, T20, I12)', "Ferr > 10% (#): ", ferr_above_10
    print '(A, T20, F12.6)', "R^2: ", r_squared
    print '(A)', "-----------------------------------------------------------------------"
    print '(A, T20, I12)', "Total Predictions: ", size(relative_error)
    print '(A, T20, F12.6)', "Total CPU Time (s): ", elapsed_time
    print '(A)', "-----------------------------------------------------------------------"

end subroutine compute_metrics

end module metrics_module