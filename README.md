# Model Evaluation and Validation
## Project: Predicting handle time, volume, and aht

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/majickdave/dji_vol/blob/master/notebooks/RIS%20Prediction%20Analysis.ipynb)


### Install

This project requires **Python** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [fbprophet](https://facebook.github.io/prophet/)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included. 

### TODO

Code is in run_model.py and must be refactored

### Run

In a terminal or command window, navigate to the top-level project directory `dji_vol/` (that contains this README) and run one of the following commands:

```bash
python code/main.py
```  

This will begin running code and provide output on each file processed

### Data

This is the volume and handle_time as well as their respective forecasts.

**Features**
1. `handle_time`: total daily talk time (seconds)
2. `aht`: daily average length of call (seconds/call)
3. `volume`: daily call count
4. `abandonment_rate`: rate of calls not recieved

