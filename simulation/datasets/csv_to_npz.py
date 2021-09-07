import pandas as pd
import numpy as np
import os

FILE_NAME = 'simple_silverbox_0_5_nonshift'
PATH = 'C:/Users/macie/Downloads/simulationcodesilverbox/' + FILE_NAME + '.csv'

if __name__ == '__main__':
    data = pd.read_csv(PATH, header=None)
    to_save = dict(inputs=np.array([data.get(i) for i in range(10)]).transpose(),
                   outputs=np.array([data.get(i) for i in range(10, 20)]).transpose())
    np.savez(os.path.join(os.path.abspath('../datasets'), FILE_NAME), **to_save)
