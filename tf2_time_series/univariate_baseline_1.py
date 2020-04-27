import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

uni_data = df['T (degC)']
uni_data.index = df['Date Time']
uni_data.head()

uni_data.plot(subplots=True)

