import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 然后再导入其他库
import torch
import torchvision

print("=" * 50)
print(" PyTorch 环境验证开始...")
print("=" * 50)

# 1. 打印版本号
print(f"✅ PyTorch 版本: {torch.__version__}")

# 2. 检查 CUDA 是否可用
if torch.cuda.is_available():
    print(f"✅ CUDA 状态: 可用 (Enabled)")
    print(f"🎮 GPU 数量: {torch.cuda.device_count()}")
    print(f"🎮 当前 GPU 名称: {torch.cuda.get_device_name(0)}")
    print(f"🔢 CUDA 版本: {torch.version.cuda}")
else:
    print("⚠️  CUDA 状态: 不可用 (将使用 CPU 运行)")
    print("💡 提示: 如果你安装了显卡驱动但显示不可用，可能需要重新安装 cuda 版本的 torch")

# 3. 创建一个简单的张量运算测试
print("\n🧪 正在进行张量计算测试...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📍 计算设备: {device}")

# 创建两个随机矩阵
x = torch.rand(5, 3).to(device)
y = torch.rand(5, 3).to(device)

# 矩阵乘法
z = torch.matmul(x, y.t()) #.t是转置

print(f"✅ 计算成功！结果矩阵形状: {z.shape}")
print(f"📊 结果前 3 行:\n{z[:3]}")

print("=" * 50)
print("🎉 验证完成！环境一切正常！")
print("=" * 50)