# test_grad_modification.py
"""测试在requires_grad=True的tensor上使用no_grad修改的行为"""

import torch

print("=" * 70)
print("🧪 测试修改requires_grad=True的tensor")
print("=" * 70)

# 测试1: 正常修改
print("\n测试1: 不使用no_grad修改")
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(f"  修改前: {x}")
x_copy = x.clone()
x_copy[0] += 10.0
print(f"  修改后clone: {x_copy}")
print(f"  原始x: {x}")

# 测试2: 使用no_grad修改
print("\n测试2: 使用no_grad修改requires_grad=True的tensor")
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(f"  修改前: {x}")
with torch.no_grad():
    x[0] += 10.0
print(f"  修改后: {x}")
print(f"  成功修改!")

# 测试3: 先detach再修改
print("\n测试3: 修改后detach")
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(f"  修改前: {x}")
with torch.no_grad():
    x[0] += 10.0
x = x.detach()
print(f"  修改后detach: {x}")

# 测试4: 模拟JSMA的修改模式
print("\n测试4: 模拟JSMA循环")
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=False)
for i in range(3):
    print(f"\n  迭代 {i+1}:")
    x.requires_grad = True
    print(f"    设置requires_grad=True: {x}")
    
    # 模拟前向传播
    y = x.sum()
    y.backward()
    
    print(f"    梯度: {x.grad}")
    
    # 修改
    with torch.no_grad():
        x[i] += 10.0
    print(f"    修改后: {x}")
    
    # detach
    x = x.detach()
    print(f"    detach后: {x}")

print("\n" + "=" * 70)
print("✅ 测试完成")
print("=" * 70)


