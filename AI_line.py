import torch
import torch.nn as nn
import torch.optim as optim

# ===================== 1. 构造数据集 =====================
# 生成输入x（随机数），形状为(100, 1)：100个样本，每个样本1个特征
x = torch.randn(100, 1)
# 生成真实标签y = 2x + 1，加少量噪声模拟真实场景
y = 2 * x + 1


# ===================== 2. 定义线性模型 =====================
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        # 定义线性层：输入维度1，输出维度1（对应y=wx+b）
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        # 前向传播：输入x经过线性层得到预测值y_pred
        return self.linear(x)


# 实例化模型
model = LinearRegression()

# ===================== 3. 定义损失函数和优化器 =====================
# 损失函数：均方误差（MSE），衡量预测值和真实值的差距
criterion = nn.MSELoss()
# 优化器：随机梯度下降（SGD），学习率0.01，更新模型的参数（w和b）
optimizer = optim.SGD(model.parameters(), lr=0.01)

# ===================== 4. 训练模型 =====================
epochs = 10000  # 训练轮数
for epoch in range(epochs):
    # 前向传播：用当前模型预测y_pred
    y_pred = model(x)

    # 计算损失：真实值y和预测值y_pred的均方误差
    loss = criterion(y_pred, y)

    # 反向传播：先清空上一轮的梯度（否则梯度会累加）
    optimizer.zero_grad()
    # 计算梯度：对损失函数求w和b的偏导数
    loss.backward()
    # 更新参数：用梯度下降更新w和b（w = w - lr*梯度，b同理）
    optimizer.step()

    # 每100轮打印一次结果，观察训练过程
    if (epoch + 1) % 100 == 0:
        # 取出模型学习到的w和b（nn.Linear的参数存在weight和bias里）
        w = model.linear.weight.item()
        b = model.linear.bias.item()
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, w: {w:.4f}, b: {b:.4f}")

# ===================== 5. 测试预测 =====================
test_x = torch.tensor([[3.0]])  # 测试输入x=3，真实y=2*3+1=7
test_y = model(test_x)
print(f"\n预测x=3时，y={test_y.item():.4f}（真实值应为7）")