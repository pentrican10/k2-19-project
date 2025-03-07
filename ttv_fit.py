from ttvfast_run import times_1, omc_1, times_2, omc_2

import matplotlib.pyplot as plt




plt.scatter(times_1,omc_1,color='orange',label='planet b lin')
plt.scatter(times_2,omc_2,color='blue', label='planet c lin')
plt.title("Linear Regression TTVs - K2-19")
plt.ylabel('TTV (days)')
plt.xlabel('Time (days)')
plt.legend()
plt.show()

