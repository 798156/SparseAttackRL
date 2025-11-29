# 清理测试脚本 - 移动到archive文件夹

Write-Host "📦 开始归档测试脚本..." -ForegroundColor Green

# 创建归档目录
$archiveRoot = "archive"
$testScripts = "$archiveRoot/test_scripts"
$oldDocs = "$archiveRoot/old_docs"
$oldAttacks = "$archiveRoot/old_attacks"
$oldTraining = "$archiveRoot/old_training"

New-Item -ItemType Directory -Path $testScripts -Force | Out-Null
New-Item -ItemType Directory -Path $oldDocs -Force | Out-Null
New-Item -ItemType Directory -Path $oldAttacks -Force | Out-Null
New-Item -ItemType Directory -Path $oldTraining -Force | Out-Null

Write-Host "✅ 创建归档目录: $archiveRoot" -ForegroundColor Cyan

# 1. 归档测试和调试脚本
Write-Host "`n1️⃣ 归档测试和调试脚本..." -ForegroundColor Yellow
$patterns = @("debug_*.py", "test_*.py", "quick_*.py", "fix_*.py", "diagnose_*.py", "verify_*.py")
$count = 0
foreach ($pattern in $patterns) {
    Get-ChildItem -Path . -Filter $pattern | ForEach-Object {
        Write-Host "   移动: $($_.Name)" -ForegroundColor Gray
        Move-Item $_.FullName -Destination $testScripts -Force
        $count++
    }
}
Write-Host "   归档了 $count 个测试脚本" -ForegroundColor Green

# 2. 归档弃用的攻击方法
Write-Host "`n2️⃣ 归档弃用的攻击方法..." -ForegroundColor Yellow
$oldAttackScripts = @(
    "one_pixel_attack.py",
    "onepixel_laptop_friendly.py",
    "optimize_onepixel.py",
    "foolbox_attacks.py",
    "foolbox_baseline_test.py",
    "pgd_l0_attack.py",
    "hybrid_attack.py"
)
$count = 0
foreach ($file in $oldAttackScripts) {
    if (Test-Path $file) {
        Write-Host "   移动: $file" -ForegroundColor Gray
        Move-Item $file -Destination $oldAttacks -Force
        $count++
    }
}
Write-Host "   归档了 $count 个弃用攻击脚本" -ForegroundColor Green

# 3. 归档RL相关脚本
Write-Host "`n3️⃣ 归档RL训练脚本..." -ForegroundColor Yellow
$rlScripts = Get-ChildItem -Path . -Filter "ppo_*.py"
$rlScripts += Get-ChildItem -Path . -Filter "sparse_attack_env*.py"
$rlScripts += Get-ChildItem -Path . -Filter "train_*rl*.py"
$count = 0
$rlScripts | ForEach-Object {
    Write-Host "   移动: $($_.Name)" -ForegroundColor Gray
    Move-Item $_.FullName -Destination $oldTraining -Force
    $count++
}
Write-Host "   归档了 $count 个RL脚本" -ForegroundColor Green

# 4. 归档训练脚本
Write-Host "`n4️⃣ 归档训练脚本..." -ForegroundColor Yellow
$trainingScripts = @(
    "train_cifar10_advanced.py",
    "train_cifar10_fast.py",
    "train_cifar10_mobilenetv2.py",
    "train_cifar10_mobilenetv2_scratch.py",
    "train_cifar10_vgg16.py"
)
$count = 0
foreach ($file in $trainingScripts) {
    if (Test-Path $file) {
        Write-Host "   移动: $file" -ForegroundColor Gray
        Move-Item $file -Destination $oldTraining -Force
        $count++
    }
}
Write-Host "   归档了 $count 个训练脚本" -ForegroundColor Green

# 5. 归档早期实验脚本
Write-Host "`n5️⃣ 归档早期实验脚本..." -ForegroundColor Yellow
$earlyScripts = @(
    "main.py",
    "main_v2.py",
    "save_day1_results.py",
    "run_100_samples_test.py",
    "run_experiment_fixed.py",
    "run_full_experiments.py",
    "run_*_experiment.py",
    "organize_*.py",
    "compare_*.py",
    "create_simple_defended_model.py",
    "display_correct_results.py",
    "download_pretrained_cifar10.py",
    "final_*.py",
    "retest_*.py",
    "statistical_analysis.py",
    "unified_baseline_test.py"
)
$count = 0
foreach ($pattern in $earlyScripts) {
    Get-ChildItem -Path . -Filter $pattern -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "   移动: $($_.Name)" -ForegroundColor Gray
        Move-Item $_.FullName -Destination $testScripts -Force
        $count++
    }
}
Write-Host "   归档了 $count 个早期脚本" -ForegroundColor Green

# 6. 归档旧文档
Write-Host "`n6️⃣ 归档旧文档..." -ForegroundColor Yellow
$docPatterns = @(
    "Day*.md",
    "今日任务*.md",
    "优化*.md",
    "快速开始*.md",
    "完整实验*.md",
    "模型训练*.md",
    "模型选择*.md",
    "模型准确率*.md",
    "方案B*.md",
    "立即开始*.md",
    "论文优化*.md",
    "论文撰写*.md",
    "问题修复*.md",
    "项目进展*.md",
    "RL训练*.md",
    "SparseAttackRL优化*.md",
    "三种攻击*.md",
    "JSMA_攻击*.md",
    "Week1调整*.md",
    "Week2_工作*.md",
    "发表路线*.md"
)
$count = 0
foreach ($pattern in $docPatterns) {
    Get-ChildItem -Path . -Filter $pattern -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "   移动: $($_.Name)" -ForegroundColor Gray
        Move-Item $_.FullName -Destination $oldDocs -Force
        $count++
    }
}
Write-Host "   归档了 $count 个旧文档" -ForegroundColor Green

# 总结
Write-Host "`n✅ 归档完成！" -ForegroundColor Green
Write-Host "📂 归档位置: $archiveRoot/" -ForegroundColor Cyan
Write-Host "   - $testScripts (测试脚本)" -ForegroundColor Gray
Write-Host "   - $oldAttacks (弃用攻击)" -ForegroundColor Gray
Write-Host "   - $oldTraining (训练脚本)" -ForegroundColor Gray
Write-Host "   - $oldDocs (旧文档)" -ForegroundColor Gray
Write-Host "`n💡 如果确认不需要，可以删除整个 archive/ 文件夹" -ForegroundColor Yellow











