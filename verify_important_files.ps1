# Verify Important Files
# Check if all important files exist

Write-Host "Checking important files..." -ForegroundColor Green

$errors = @()
$warnings = @()

# 1. Paper Documents
Write-Host "`n[1] Checking paper documents..." -ForegroundColor Yellow
$papers = @(
    "论文草稿-稀疏对抗攻击综合研究.md",
    "latex_paper/main.tex",
    "latex_paper/main_chinese_simple.tex",
    "latex_paper/references.bib",
    "README.md"
)
foreach ($file in $papers) {
    if (Test-Path $file) {
        Write-Host "   [OK] $file" -ForegroundColor Green
    } else {
        Write-Host "   [MISSING] $file" -ForegroundColor Red
        $errors += $file
    }
}

# 2. Core Models
Write-Host "`n[2] Checking core models..." -ForegroundColor Yellow
$models = @(
    "cifar10_resnet18_best.pth",
    "cifar10_vgg16_best.pth",
    "cifar10_mobilenetv2_best.pth",
    "models/cifar10/Linf/Standard.pt",
    "models/cifar10/Linf/Engstrom2019Robustness.pt",
    "models/cifar10/Linf/Rice2020Overfitting.pt"
)
foreach ($file in $models) {
    if (Test-Path $file) {
        $size = (Get-Item $file).Length / 1MB
        Write-Host "   [OK] $file ($([math]::Round($size, 2)) MB)" -ForegroundColor Green
    } else {
        Write-Host "   [MISSING] $file" -ForegroundColor Red
        $errors += $file
    }
}

# 3. Core Data
Write-Host "`n[3] Checking core data..." -ForegroundColor Yellow
$dataFolders = @(
    "results/complete_baseline",
    "results/analysis_5methods",
    "results/parameter_sensitivity",
    "results/multi_defense_models",
    "results/complete_defense_comparison",
    "results/failure_analysis",
    "results/class_analysis",
    "results/adversarial_visualization",
    "results/query_efficiency"
)
foreach ($folder in $dataFolders) {
    if (Test-Path $folder) {
        $fileCount = (Get-ChildItem -Path $folder -File).Count
        Write-Host "   [OK] $folder ($fileCount files)" -ForegroundColor Green
    } else {
        Write-Host "   [MISSING] $folder" -ForegroundColor Red
        $errors += $folder
    }
}

# 4. Core Scripts
Write-Host "`n[4] Checking core scripts..." -ForegroundColor Yellow
$scripts = @(
    "complete_baseline_5methods.py",
    "analyze_parameter_sensitivity.py",
    "test_multiple_defense_models.py",
    "create_complete_defense_comparison.py",
    "analyze_failure_cases.py",
    "analyze_class_specific_asr.py",
    "generate_confusion_matrices.py",
    "visualize_adversarial_examples.py",
    "analyze_query_efficiency.py",
    "jsma_attack.py",
    "sparsefool_attack.py",
    "greedy_attack.py",
    "pixel_gradient_attack.py",
    "random_sparse_attack.py",
    "attack_adapters.py",
    "load_trained_model.py",
    "dataset_loader.py",
    "evaluation_metrics.py"
)
foreach ($file in $scripts) {
    if (Test-Path $file) {
        Write-Host "   [OK] $file" -ForegroundColor Green
    } else {
        Write-Host "   [WARNING] Missing: $file" -ForegroundColor Yellow
        $warnings += $file
    }
}

# 5. Statistics
Write-Host "`n[5] Counting figures..." -ForegroundColor Yellow
$pdfCount = (Get-ChildItem -Path results -Filter "*.pdf" -Recurse -ErrorAction SilentlyContinue).Count
$pngCount = (Get-ChildItem -Path results -Filter "*.png" -Recurse -ErrorAction SilentlyContinue).Count
Write-Host "   PDF figures: $pdfCount" -ForegroundColor Cyan
Write-Host "   PNG figures: $pngCount" -ForegroundColor Cyan

# Summary
Write-Host "`n============================================================" -ForegroundColor Cyan
if ($errors.Count -eq 0) {
    Write-Host "SUCCESS! All important files are present!" -ForegroundColor Green
} else {
    Write-Host "ERROR! Found $($errors.Count) missing critical files!" -ForegroundColor Red
    Write-Host "Missing files:" -ForegroundColor Red
    $errors | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
}

if ($warnings.Count -gt 0) {
    Write-Host "`nWARNING! Found $($warnings.Count) missing optional files" -ForegroundColor Yellow
    Write-Host "Suggested to check:" -ForegroundColor Yellow
    $warnings | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
}

Write-Host "============================================================" -ForegroundColor Cyan
