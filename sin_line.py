import numpy as np
import matplotlib.pyplot as plt

# 生成数据
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 画图
plt.plot(x, y)
plt.show()