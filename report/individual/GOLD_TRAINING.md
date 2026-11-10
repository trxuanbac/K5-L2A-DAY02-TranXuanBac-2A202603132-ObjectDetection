# Thí nghiệm huấn luyện theo gold đã chốt

Theo yêu cầu tiếp theo của học viên, dùng trực tiếp bộ gold được cung cấp làm nhãn cho một thí nghiệm **gold_reference** riêng. Đây không phải rework nhãn học viên, không chứng minh nhãn học viên đạt gold QC và không thay kết quả F1 = 0,4926 đã lưu.

## Dữ liệu và lớp

- Train: 50 ảnh `drive_001`–`drive_050`, nhãn từ `gold_train_labels.rar`; giữ nguyên byte của 50 file gold, tổng 653 box.
- Validation: 10 ảnh `drive_051`–`drive_060` và nhãn eval gốc; không đưa eval vào train.
- Class ID thực tế dùng chung giữa gold và eval: **0=car, 1=van, 2=truck, 3=bus**. Đã đối chiếu trực quan bus trong ảnh train `drive_008` và ảnh eval `drive_052`, `drive_058`.
- Histogram train theo raw ID: car 533, van 46, truck 20, bus 54. Eval: car 94, van 5, truck 2, bus 12.
- Phạm vi vật thể giữ nguyên theo các box của gold, không tự bổ sung hoặc xóa box để khớp guideline trước đây.

`data.yaml`, nhãn học viên và CVAT giữ schema của bài gốc. Chỉ cấu hình runtime `outputs/gold_training/data_gold.yaml` dùng bảng ID của thí nghiệm này. `tools/train_gold.py` tạo dataset runtime riêng và kiểm tra hash để đảm bảo không vô tình lấy nhãn học viên hoặc sửa gold/eval.

## Chạy lại

Cần Python có `ultralytics`, `torch`, `Pillow`, `pyyaml`, checkpoint pretrained `yolo26n.pt`, và thư mục gold local. Từ thư mục gốc repo:

```bash
python3 tools/train_gold.py \
  --gold-labels gold_train_labels \
  --weights /path/to/yolo26n.pt \
  --epochs 50 --batch 8 --device cpu
```

Thông số: ảnh 640, seed 20260909, AMP tắt, workers 0. Lần chạy đầu dùng Apple MPS; khi MPS ngừng cập nhật ở epoch 5, đã lưu checkpoint và resume trên CPU từ epoch 6, giữ optimizer/EMA và mục tiêu tổng cộng 50 epoch. Lệnh trên dùng CPU để tránh lặp lại sự cố MPS. Đây là phiên chạy local, không phải phiên Google Colab.

MPS cảnh báo một phép `index_put_with_accumulate_mps` chưa có triển khai deterministic. Có đặt seed nhưng không cam kết tái lập từng bit trên MPS.

## Đầu ra

- `outputs/gold_training/manifest.json`: nguồn nhãn, hash, phiên bản và tham số.
- `outputs/gold_training/status.json`, `train.log`: tiến độ và log thực tế.
- `outputs/gold_training/yolo26_runs/`: checkpoint và các đồ thị.
- `outputs/gold_training/model_eval_metrics.json`: chỉ số trên validation sau khi train.
- `outputs/gold_training/eval_predictions/`: 10 ảnh dự đoán.
- `outputs/gold_training/day2_gold_outputs.zip`: gói kết quả local.
- `report/individual/assets/`: bản các chỉ số, đồ thị và prediction để đưa vào báo cáo sau khi chạy hoàn tất.
- `notebooks/day2_gold_results.ipynb`: notebook đọc các artifact thật và hiển thị kết quả; không giả lập việc train qua Colab.

Gold train, dataset runtime, checkpoint `.pt` và ảnh `train_batch*` có chứa box gold được giữ local. Không đưa các nội dung đó lên GitHub. Các ảnh prediction công khai là kết quả của model trên 10 ảnh eval.

Eval đồng thời làm validation chọn best checkpoint, nên các chỉ số không đại diện cho một test set độc lập. Xem [báo cáo cá nhân](2A202603132_TranXuanBac.md) để biết trạng thái chạy và kết quả cuối cùng.

## Kết quả đã chạy

Đã hoàn tất đúng 50 epoch (5 MPS + 45 CPU), đánh giá riêng best.pt trên 10 ảnh eval và xuất 10 prediction.

- Precision: 0.518141.
- Recall: 0.646748.
- mAP50: 0.645446.
- mAP50–95: 0.451602.

[Chỉ số model](assets/model_eval_metrics_gold.json) và [đồ thị train](assets/yolo26_runs/vehicle_gold/results.png). Notebook kết quả đọc artifact thật của phiên local, không phải phiên huấn luyện Colab.

## Cập nhật sau thí nghiệm

Sau phiên gold_reference, nhãn student đã được rework và chấm riêng: F1 0,9508, mean IoU 0,8488, đạt gate. Xem CP5 trong báo cáo cá nhân. Phiên model và bốn chỉ số ở tài liệu này vẫn thuộc bộ 653 box gold; chưa phải model train từ 607 box student sau rework.
