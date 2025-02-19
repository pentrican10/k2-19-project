from ttvfast_run import t_1, residuals1, t_2, residuals2

import matplotlib.pyplot as plt




plt.scatter(t_1,residuals1,color='orange',label='planet b lin')
plt.scatter(t_2,residuals2,color='blue', label='planet c lin')
plt.title("Linear Regression TTVs - K2-19")
plt.ylabel('TTV (days)')
plt.xlabel('Time (days)')
plt.legend()
plt.show()

