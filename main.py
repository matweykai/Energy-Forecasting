from exl_functions import get_data, write_data, plot_results
from prediction_methods import get_ARMA_prediction, get_exp_autoreg_prediction, get_ssa_prediction, get_regr_prediction


if __name__ == '__main__':
    cur_file = 'data/Prognozy.xlsx'

    # Loading data from Excel file
    data = get_data(cur_file, 1)

    # Making prediction with linear regression
    regr_prediction = get_regr_prediction(data, 24)
    write_data(cur_file, 2, regr_prediction)

    # Making ARMA prediction
    arma_prediction = get_ARMA_prediction(data, 3, 0, 2, 24)
    write_data(cur_file, 3, arma_prediction)

    # Making prediction with exponential autoregressive model
    exp_prediction = get_exp_autoreg_prediction(data, 24)
    write_data(cur_file, 4, exp_prediction)

    # Making prediction with the SSA method
    ssa_prediction = get_ssa_prediction(data, 24)
    write_data(cur_file, 5, ssa_prediction)

    # Plotting results in Excel file
    plot_results(cur_file)
