# 安全清理脚本 - 删除确定无用的文件
# 预计节省空间: 5-7GB

Write-Host "🧹 开始安全清理..." -ForegroundColor Green

# 创建备份记录
$logFile = "cleanup_log_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
Write-Host "📝 记录到: $logFile" -ForegroundColor Cyan

# 统计函数
function Get-FileSize {
    param($files)
    ($files | Measure-Object -Property Length -Sum).Sum / 1GB
}

# 1. 删除训练checkpoint
Write-Host "`n1️⃣ 删除训练checkpoint..." -ForegroundColor Yellow
$checkpoints = Get-ChildItem -Path . -Filter "checkpoint_*.pth"
$size1 = Get-FileSize $checkpoints
Write-Host "   找到 $($checkpoints.Count) 个文件，约 $([math]::Round($size1, 2)) GB" -ForegroundColor Gray
$checkpoints | ForEach-Object { 
    "Deleted: $($_.Name)" | Add-Content $logFile
    Remove-Item $_.FullName -Force 
}

# 2. 删除重复模型
Write-Host "`n2️⃣ 删除重复模型..." -ForegroundColor Yellow
$duplicates = @(
    "cifar10_resnet18_888.pth",
    "cifar10_resnet18_best_888.pth",
    "cifar10_resnet18.pth",
    "cifar10_vgg16.pth",
    "cifar10_mobilenetv2.pth"
)
foreach ($file in $duplicates) {
    if (Test-Path $file) {
        $item = Get-Item $file
        $size = $item.Length / 1MB
        Write-Host "   删除: $file ($([math]::Round($size, 2)) MB)" -ForegroundColor Gray
        "Deleted: $file" | Add-Content $logFile
        Remove-Item $file -Force
    }
}

# 3. 删除RL模型
Write-Host "`n3️⃣ 删除RL模型..." -ForegroundColor Yellow
$rlModels = Get-ChildItem -Path . -Filter "ppo_sparse*.zip"
$rlModels | ForEach-Object {
    Write-Host "   删除: $($_.Name)" -ForegroundColor Gray
    "Deleted: $($_.Name)" | Add-Content $logFile
    Remove-Item $_.FullName -Force
}

if (Test-Path "models/ppo_resnet18_v3.zip") {
    Remove-Item "models/ppo_resnet18_v3.zip" -Force
    "Deleted: models/ppo_resnet18_v3.zip" | Add-Content $logFile
}
if (Test-Path "models/ppo_resnet18_v3_simple.zip") {
    Remove-Item "models/ppo_resnet18_v3_simple.zip" -Force
    "Deleted: models/ppo_resnet18_v3_simple.zip" | Add-Content $logFile
}

# 4. 删除过时结果文件夹
Write-Host "`n4️⃣ 删除过时结果文件夹..." -ForegroundColor Yellow
$oldResults = @(
    "results/final",
    "results/final_baseline",
    "results/foolbox_baseline",
    "results/full_experiments",
    "results/unified_baseline",
    "results/v2",
    "results/v2_fixed",
    "results/week1_day1",
    "results/week1_day2",
    "results/week1_day5",
    "results/plots"
)
foreach ($dir in $oldResults) {
    if (Test-Path $dir) {
        $size = Get-FileSize (Get-ChildItem -Path $dir -Recurse -File)
        Write-Host "   删除文件夹: $dir ($([math]::Round($size, 3)) GB)" -ForegroundColor Gray
        "Deleted folder: $dir" | Add-Content $logFile
        Remove-Item $dir -Recurse -Force
    }
}

# 5. 删除实验日志
Write-Host "`n5️⃣ 删除实验日志..." -ForegroundColor Yellow
$logs = Get-ChildItem -Path results -Filter "experiment_*.txt" -ErrorAction SilentlyContinue
$logs | ForEach-Object {
    Write-Host "   删除: $($_.Name)" -ForegroundColor Gray
    "Deleted: $($_.Name)" | Add-Content $logFile
    Remove-Item $_.FullName -Force
}

# 6. 删除LaTeX临时文件
Write-Host "`n6️⃣ 删除LaTeX临时文件..." -ForegroundColor Yellow
$latexTemp = @(
    "latex_paper/*.log",
    "latex_paper/*.aux",
    "latex_paper/*.synctex*",
    "latex_paper/main_chinese.tex",
    "latex_paper/main_chinese_complete.tex"
)
foreach ($pattern in $latexTemp) {
    Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "   删除: $($_.Name)" -ForegroundColor Gray
        "Deleted: $($_.Name)" | Add-Content $logFile
        Remove-Item $_.FullName -Force
    }
}

# 7. 删除Python缓存
Write-Host "`n7️⃣ 删除Python缓存..." -ForegroundColor Yellow
if (Test-Path "__pycache__") {
    Remove-Item "__pycache__" -Recurse -Force
    "Deleted: __pycache__/" | Add-Content $logFile
}

Write-Host "`n✅ 安全清理完成！" -ForegroundColor Green
Write-Host "📋 详细日志: $logFile" -ForegroundColor Cyan
Write-Host "💾 建议运行 'cleanup_test_scripts.ps1' 继续清理测试脚本" -ForegroundColor Yellow











