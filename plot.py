import matplotlib.pyplot as plt
import numpy as np


x = [0, 0.01, 0.05, 0.1, 0.5,1]
ce = [3.67,4.69,4.14,3.85,3.74,3.87]
plt.title('T5-B CE with different sized OST')
plt.plot(ce, color = 'blue')
plt.xlabel('data portion(50%=0.5)')
plt.ylabel('BLEU2')
plt.xticks(np.arange(len(x)),x)

plt.show()
plt.savefig('ce.png')