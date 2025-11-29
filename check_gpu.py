# check_gpu.py
"""
GPU配置检查脚本
运行此脚本确认PyTorch能否正确识别你的RTX 4060
"""

import torch
import sys

print("=" * 60)
print("🔍 GPU配置检查")
print("=" * 60)

# 1. 检查CUDA是否可用
print(f"\n1. CUDA是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # 2. CUDA版本
    print(f"2. CUDA版本: {torch.version.cuda}")
    
    # 3. GPU数量
    print(f"3. GPU数量: {torch.cuda.device_count()}")
    
    # 4. 当前GPU信息
    print(f"4. 当前GPU: {torch.cuda.get_device_name(0)}")
    
    # 5. GPU内存
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"5. GPU显存: {total_memory:.2f} GB")
    
    # 6. 测试GPU计算
    print("\n6. 测试GPU计算...")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.mm(x, y)
        print("   ✅ GPU计算正常！")
        
        # 测试速度
        import time
        
        # CPU测试
        x_cpu = torch.randn(2000, 2000)
        y_cpu = torch.randn(2000, 2000)
        start = time.time()
        for _ in range(10):
            z_cpu = torch.mm(x_cpu, y_cpu)
        cpu_time = time.time() - start
        
        # GPU测试
        x_gpu = torch.randn(2000, 2000).cuda()
        y_gpu = torch.randn(2000, 2000).cuda()
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            z_gpu = torch.mm(x_gpu, y_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        print(f"\n📊 速度对比（2000x2000矩阵乘法 x10次）:")
        print(f"   CPU耗时: {cpu_time:.4f}秒")
        print(f"   GPU耗时: {gpu_time:.4f}秒")
        print(f"   🚀 GPU加速: {cpu_time/gpu_time:.2f}x")
        
    except Exception as e:
        print(f"   ❌ GPU计算失败: {e}")
    
    print("\n" + "=" * 60)
    print("✅ GPU配置正常！可以运行实验了！")
    print("=" * 60)
    
else:
    print("\n❌ CUDA不可用！")
    print("\n可能的原因：")
    print("1. 没有安装CUDA版本的PyTorch")
    print("2. NVIDIA驱动未安装或版本过旧")
    print("3. CUDA toolkit未正确安装")
    print("\n解决方案：")
    print("重新安装PyTorch (CUDA版本):")
    print("pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    print("\n或访问: https://pytorch.org/get-started/locally/")
    print("=" * 60)
    sys.exit(1)

