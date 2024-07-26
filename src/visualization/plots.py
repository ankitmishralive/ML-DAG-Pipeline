

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


current_path = Path(__file__)
root=current_path.parent.parent.parent 
data_path = root / 'data' / 'raw'/'extracted'

df = pd.read_csv(data_path/'corpus.csv')
df['ph'].hist()
plt.title('Distribution of ph Column')
plt.savefig('histogram_distribution.png')  
sns.boxplot(x=df['ph'])
plt.title('Boxplot of ph Column')
plt.show()




