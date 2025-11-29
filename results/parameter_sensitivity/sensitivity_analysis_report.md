# Parameter Sensitivity Analysis Report

**Generated:** 2025-11-06 11:29:21  
**Model:** resnet18  
**Samples per configuration:** 50

---

## Executive Summary

This report analyzes the sensitivity of 5 sparse adversarial attack methods to their key parameters.
The goal is to identify optimal parameters and assess method stability.

---

## Methods and Parameters Tested

### JSMA
- **Parameter:** max_pixels
- **Values:** [5, 10, 15, 20]
- **Fixed params:** {'theta': 0.2, 'max_iterations': 100}

### SparseFool
- **Parameter:** max_iter
- **Values:** [10, 20, 30, 50]
- **Fixed params:** {'overshoot': 0.02, 'lambda_': 3.0}

### Greedy
- **Parameter:** max_pixels
- **Values:** [5, 10, 15, 20]
- **Fixed params:** {'alpha': 0.1, 'max_iterations': 100}

### PixelGrad
- **Parameter:** max_pixels
- **Values:** [5, 10, 15, 20]
- **Fixed params:** {'alpha': 0.2, 'beta': 0.9}

### RandomSparse
- **Parameter:** max_pixels
- **Values:** [5, 10, 15, 20]
- **Fixed params:** {'perturbation_size': 0.2, 'max_attempts': 50}

---

## Detailed Results

### JSMA

| Parameter | ASR (%) | Avg L0 | Avg L2 | Avg Time (s) |
|-----------|---------|--------|--------|-------------|
| 5 | 2.0 | 1.00 | 0.2000 | 0.085 |
| 10 | 2.0 | 1.00 | 0.2000 | 0.071 |
| 15 | 2.0 | 1.00 | 0.2000 | 0.072 |
| 20 | 6.0 | 12.67 | 0.6397 | 1.085 |

### SparseFool

| Parameter | ASR (%) | Avg L0 | Avg L2 | Avg Time (s) |
|-----------|---------|--------|--------|-------------|
| 10 | 8.0 | 1.75 | 2.3597 | 0.092 |
| 20 | 14.0 | 5.00 | 2.6391 | 0.153 |
| 30 | 14.0 | 5.00 | 2.6391 | 0.156 |
| 50 | 20.0 | 12.60 | 3.3355 | 0.333 |

### Greedy

| Parameter | ASR (%) | Avg L0 | Avg L2 | Avg Time (s) |
|-----------|---------|--------|--------|-------------|
| 5 | 2.0 | 1.00 | 0.5015 | 0.038 |
| 10 | 8.0 | 5.75 | 1.1322 | 0.143 |
| 15 | 12.0 | 8.00 | 1.3408 | 0.199 |
| 20 | 16.0 | 10.12 | 1.5113 | 0.233 |

### PixelGrad

| Parameter | ASR (%) | Avg L0 | Avg L2 | Avg Time (s) |
|-----------|---------|--------|--------|-------------|
| 5 | 4.0 | 1.50 | 1.9865 | 0.045 |
| 10 | 6.0 | 3.33 | 1.7816 | 0.086 |
| 15 | 6.0 | 3.33 | 1.7816 | 0.079 |
| 20 | 6.0 | 3.33 | 1.7816 | 0.077 |

### RandomSparse

| Parameter | ASR (%) | Avg L0 | Avg L2 | Avg Time (s) |
|-----------|---------|--------|--------|-------------|
| 5 | 6.0 | 11.00 | 4.1498 | 0.156 |
| 10 | 4.0 | 22.50 | 5.2788 | 0.086 |
| 15 | 10.0 | 30.00 | 5.0950 | 0.144 |
| 20 | 12.0 | 42.00 | 6.1817 | 0.127 |

---

## Key Findings

### Optimal Parameters

- **JSMA:** max_pixels=20 (ASR=6.0%)
- **SparseFool:** max_iter=50 (ASR=20.0%)
- **Greedy:** max_pixels=20 (ASR=16.0%)
- **PixelGrad:** max_pixels=10 (ASR=6.0%)
- **RandomSparse:** max_pixels=20 (ASR=12.0%)

### Stability Ranking

Methods ranked by parameter sensitivity (lower variance = more stable):

1. **PixelGrad** - Std: 0.87%, Variance: 0.75
2. **JSMA** - Std: 1.73%, Variance: 3.00
3. **RandomSparse** - Std: 3.16%, Variance: 10.00
4. **SparseFool** - Std: 4.24%, Variance: 18.00
5. **Greedy** - Std: 5.17%, Variance: 26.75

---

## Recommendations

1. **Most Stable Method:** PixelGrad shows the lowest sensitivity to parameter changes.
2. **Optimal Parameters:** Use the parameters identified above for best ASR.
3. **Trade-offs:** Consider L0 norm and time when choosing parameters.

---

## Visualizations

- `asr_sensitivity_curves.pdf` - ASR vs parameter value
- `l0_sensitivity_curves.pdf` - L0 norm vs parameter value
- `time_sensitivity_curves.pdf` - Time vs parameter value
- `comprehensive_heatmap.pdf` - All metrics heatmap

---

*Report generated: 2025-11-06 11:29:21*
