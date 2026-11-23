# Results

## Headline

| Metric | Value |
| --- | --- |
| Scenes | 120 |
| Strict success | **99.17%** |
| Mean IoU | 0.8953 |
| Median corner error | 0.0196 |
| Mean runtime | 75.97 ms |

## By condition

| condition   |   samples |   success_rate |   corner_success_rate |   iou_success_rate |   median_corner_error |   mean_iou |   mean_runtime_ms |
|:------------|----------:|---------------:|----------------------:|-------------------:|----------------------:|-----------:|------------------:|
| clean       |        20 |          1.000 |                 1.000 |              1.000 |                 0.020 |      0.899 |            75.916 |
| rotate      |        20 |          1.000 |                 1.000 |              1.000 |                 0.020 |      0.898 |            75.551 |
| blur        |        20 |          1.000 |                 1.000 |              1.000 |                 0.019 |      0.906 |            73.940 |
| noise       |        20 |          0.950 |                 0.950 |              0.950 |                 0.020 |      0.872 |            78.767 |
| shadow      |        20 |          1.000 |                 1.000 |              1.000 |                 0.019 |      0.900 |            76.278 |
| lowcontrast |        20 |          1.000 |                 1.000 |              1.000 |                 0.020 |      0.896 |            75.345 |

## Robustness

The full sweep is in `reports/robustness.csv`. Success degrades gradually with
blur, noise and rotation; the sharpest drop occurs under heavy shadow, where the
brightness mask and page boundary merge.

## Figures

| Figure | Content |
| --- | --- |
| 01_dataset_grid.png | Nine synthetic scenes |
| 02_detection_examples.png | Detections and rectified pages |
| 03_success_by_condition.png | Success rate per condition |
| 04_corner_error_distribution.png | Corner error histogram |
| 05_iou_vs_error.png | Geometry agreement scatter |
| 06_robustness_blur_noise.png | Blur and noise sweeps |
| 07_robustness_rotation_shadow.png | Rotation and shadow sweeps |
| 08_failure_cases.png | Predicted vs ground-truth failures |
