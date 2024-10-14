import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.misc import derivative
from sympy import *
from galaxy import *

def read_dat(filename):
    head = []
    with open(filename, 'r') as f:
        for i in range(3):
            head.append(next(f).strip()[2:].split('\t'))

    frame = pd.read_csv(filename, sep='\t', comment='#', names=head[1])

    distance = float(head[0][0].split()[2])
    distance_units = head[0][0].split()[3]

    units = head[2]

    data = {
        'frame': frame,
        'feature_names': head[1],
        'feature_units': units,
        'distance': distance,
        'distance_units': distance_units
    }

    return data

example_data = read_dat("Rotmass\Rotmass\IC4202_rotmass.dat")
example_df = example_data["frame"]

IC = Galaxy(example_df)
rad = IC.rad
print(rad)
# v_inter = IC.v_inter
# vobs = IC.vobs


# rc = sns.lineplot(x=rad, y=v_inter, label = "Model")
# sns.scatterplot(x=rad, y=vobs, label = "Vobs data")
# rc.set_xlabel("Radius")
# rc.set_ylabel("Velocity")
# rc.legend()
# plt.show()