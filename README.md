# CambridgeTemperatureNotebooks

![Lifecycle
](https://img.shields.io/badge/lifecycle-experimental-orange.svg?style=flat)
![Python
](https://img.shields.io/badge/Python-blue.svg?style=flat)

Time series and other models for Cambridge UK temperature forecasts in python

If you like CambridgeTemperatureNotebooks, give it a star, or fork it and
contribute!

Summary of single step-ahead predictions using simple Long Short Term Memory
model with 12 hours of lagged variables plus test time augmentation:

![](https://github.com/makeyourownmaker/CambridgeTemperatureNotebooks/blob/main/figures/LSTM_24lags_1stepahead_TTA.01.png)

These predictions are for separate test data from 2019.
RMSE is 7.26 and MAE is 3.71.


## Installation/Usage

Required:
 * [python](https://www.python.org/)
 * [Jupyter](https://jupyter.org/)
 * [pandas](https://pandas.pydata.org/)
 * [numpy](http://numpy.org/)
 * [matplotlib](http://matplotlib.org/)
 * [seaborn](https://seaborn.pydata.org/)
 * [tensorflow >= 2.2](tensorflow.org)
 * [statsmodels](https://www.statsmodels.org/stable/index.html)

To install the python packages:
```sh
pip install -r requirements.txt
```

After the above dependencies have been installed either,
 * clone the repository and open the notebook(s) in a local installation of Jupyter

or,
 * try notebook(s) remotely
   * MLP, FCN, ResNet, LSTM for temperature forecasts
     * [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/makeyourownmaker/CambridgeTemperatureNotebooks/blob/main/notebooks/keras_mlp_fcn_resnet_time_series.ipynb) - editable
     * [![Binder](https://binder.pangeo.io/badge_logo.svg)](https://mybinder.org/v2/gh/makeyourownmaker/CambridgeTemperatureNotebooks/main?filepath=notebooks%2Fkeras_mlp_fcn_resnet_time_series.ipynb) - editable
     * View on [NBViewer](https://nbviewer.jupyter.org/github/makeyourownmaker/CambridgeTemperatureNotebooks/blob/main/notebooks/keras_mlp_fcn_resnet_time_series.ipynb)
     * View on [GitHub](https://github.com/makeyourownmaker/CambridgeTemperatureNotebooks/blob/main/notebooks/keras_mlp_fcn_resnet_time_series.ipynb)
   * 2021 baseline forecasts
     * [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/makeyourownmaker/CambridgeTemperatureNotebooks/blob/main/notebooks/cammet_baselines_2021.ipynb) - editable
     * [![Binder](https://binder.pangeo.io/badge_logo.svg)](https://mybinder.org/v2/gh/makeyourownmaker/CambridgeTemperatureNotebooks/main?filepath=notebooks%2Fcammet_baselines_2021.ipynb) - editable
     * View on [NBViewer](https://nbviewer.jupyter.org/github/makeyourownmaker/CambridgeTemperatureNotebooks/blob/main/notebooks/cammet_baselines_2021.ipynb)
     * View on [GitHub](https://github.com/makeyourownmaker/CambridgeTemperatureNotebooks/blob/main/notebooks/cammet_baselines_2021.ipynb)


## Details

See my
[time series and other models for Cambridge UK temperature forecasts in R repository](https://github.com/makeyourownmaker/CambridgeTemperatureModel)
for a detailed explanation of the data (including cleaning), baseline models, 
daily and yearly seasonality descriptions plus R prophet model.  Assumptions
and limitations are covered in the above repository and will not be repeated
here.  Additional exploratory data analysis is available in my
[Cambridge University Computer Laboratory Weather Station R Shiny repository](https://github.com/makeyourownmaker/ComLabWeatherShiny).

My primary interest is in "now-casting" or forecasts within the 
next 1 to 2 hours.  This is because I live close to the data source and 
the [UK met office](https://www.metoffice.gov.uk/) only update their public 
facing forecasts every 2 hours.

### MLP, FCN, ResNet, LSTM for temperature forecasts

There are 10 years of training data (mid-2008 to 2017 inclusive) plus
validation data from 2018 and test data from 2019.

I use the following neural network architectures to make temperature forecasts:
 * Multi-layer perceptron (MLP)
 * Fully convolutional network (FCN)
 * Residual network (ResNet)
 * Long short term memory (LSTM)

The [mixup method](https://arxiv.org/abs/1710.09412)
is used to counteract the categorical wind bearing measurements.

More details are included in the
[keras MLP, FCN, ResNet, LSTM time series notebook](https://github.com/makeyourownmaker/CambridgeTemperatureNotebooks/blob/main/notebooks/keras_mlp_fcn_resnet_time_series.ipynb).

### 2021 baseline forecasts

There are over 10 years of data (mid-2008 to early-2021 inclusive).

I compare forecasts from univariate and multivariate methods in the
[statsmodels](https://www.statsmodels.org/stable/index.html) package
to establish reasonable baselines results.

Methods include:
 * persistent
 * simple exponential smoothing
 * Holt Winter's exponential smoothing
 * vector autoregression
 * SARIMAX

More details are included in the
[2021 baseline forecasts notebook](https://github.com/makeyourownmaker/CambridgeTemperatureNotebooks/blob/main/notebooks/cammet_baselines_2021.ipynb).


## Roadmap

 * Add standard deviations to MSE, MAE and MAPE values
 * Examine [Global Forecast System](https://www.ncdc.noaa.gov/data-access/model-data/model-datasets/global-forcast-system-gfs) (GFS) weather model
   * runs four times a day, produces forecasts up to 16 days in advance
   * data is available for free in the public domain
   * model serves as the basis for the forecasts of numerous services
   * potentially use as additional exogeneous variables
 * Improve tensorflow/keras models
   * optimise architectures and other hyperparameters
   * see future work section in the [keras MLP, FCN, ResNet, LSTM time series notebook](https://github.com/makeyourownmaker/CambridgeTemperatureNotebooks/blob/main/notebooks/keras_mlp_fcn_resnet_time_series.ipynb).
 * Check temporal fusion transformers performance
   * [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363)


## Contributing

Pull requests are welcome.  For major changes, please open an issue first to discuss what you would like to change.


## Alternatives

* [UK Met Office](https://metoffice.gov.uk/)
* [Cambridge University Computer Laboratory Weather Station R Shiny Web App](https://github.com/makeyourownmaker/ComLabWeatherShiny)
* [Forecasting surface temperature based on latitude, longitude, day of year and hour of day](https://github.com/makeyourownmaker/ParametricWeatherModel)
* [Time series and other models for Cambridge UK temperature forecasts in R](https://github.com/makeyourownmaker/CambridgeTemperatureModel)


## License

[GPL-2](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
