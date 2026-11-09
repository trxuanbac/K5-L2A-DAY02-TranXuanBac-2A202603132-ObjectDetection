# Báo cáo cá nhân — Trần Xuân Bắc — Ngày 2 Object Detection

## 1. Thông tin học viên

- Họ tên: Trần Xuân Bắc.
- MSSV: 2A202603132.
- Lớp: L2A.
- Email: chưa bổ sung.
- Repo bài làm: https://github.com/trxuanbac/K5-L2A-DAY02-TranXuanBac-2A202603132-ObjectDetection.
- [Mở notebook trên Colab](https://colab.research.google.com/github/trxuanbac/K5-L2A-DAY02-TranXuanBac-2A202603132-ObjectDetection/blob/main/notebooks/day2_annotation_to_yolo.ipynb). Chưa có phiên Colab đã chạy.
- [Notebook kết quả thí nghiệm gold](../../notebooks/day2_gold_results.ipynb): đọc artifact của phiên train local; không phải phiên train Colab.

## 2. Schema và mini guideline

Áp dụng [GUIDELINE_MINI.md](../../GUIDELINE_MINI.md): `car=0`, `truck=1`, `bus=2`, `van=3`; bao sát phần phương tiện nhìn thấy, giới hạn ở mép ảnh và bỏ vật thể không đủ rõ để xác định lớp.

Ba ca đã ghi trong guideline: xe bị che ở `drive_009`, xe buýt cắt khung ở `drive_029`, phương tiện nhỏ/mờ ở `drive_001`. Đối chiếu gold phát hiện ID lớp và phạm vi xe đỗ chưa khớp schema/guideline; xem CP5.

## 3. Annotation và CP3 — QC có AI hỗ trợ

- Có 50/50 file YOLO, tương ứng đúng 50 ảnh train.
- Sau CP3: **1.182 box**, gồm car 942, truck 31, bus 103, van 106.
- `python3 tools/check_yolo_labels.py`: **ĐẠT cấu trúc 50/50**.
- Chọn ngẫu nhiên 10 ảnh bằng `random.Random(20261109).sample(sorted(train_images), 10)`; giữ seed để tái lập.
- Codex quan sát cả ảnh có box và ảnh gốc; đây là lượt kiểm tra của AI, **học viên chưa xác nhận đã tự rà bằng mắt**.

| Ảnh ngẫu nhiên | Nội dung đã rà và kết quả |
|---|---|
| [drive_001.jpg](assets/self_qc/drive_001.jpg) | Box xe đỗ, van và xe bị cột che; giữ quy tắc phần nhìn thấy. |
| [drive_004.jpg](assets/self_qc/drive_004.jpg) | Bỏ 1 box car trên vùng xe máy/đèn pha mơ hồ ở phía trên trái; không đủ dấu hiệu của ô tô. |
| [drive_009.jpg](assets/self_qc/drive_009.jpg) | Rà riêng từng bus, truck và van trong cảnh đông xe; box bus trước bao trọn phần thân nhìn thấy. |
| [drive_012.jpg](assets/self_qc/drive_012.jpg) | Bỏ 1 box car nhận nhầm xe máy ở mép trái; giữ các box van và bus riêng. |
| [drive_019.jpg](assets/self_qc/drive_019.jpg) | Chuyển xe thân dài có cửa sổ bên ở nửa trái từ car sang van; xe cứu hộ giữ truck. |
| [drive_024.jpg](assets/self_qc/drive_024.jpg) | Kiểm tra van chồng lấp, xe buýt và xe ba bánh; xe ba bánh không được gán. |
| [drive_028.jpg](assets/self_qc/drive_028.jpg) | Rà xe đỗ hai bên, van trắng bên trái và xe bị cắt mép dưới. |
| [drive_032.jpg](assets/self_qc/drive_032.jpg) | Rà box xe có đèn pha, chuyển động mờ và car bị cắt ở mép dưới. |
| [drive_040.jpg](assets/self_qc/drive_040.jpg) | Rà bus dài, xe tải chở hàng và các van; không gán xe đạp/xe máy. |
| [drive_050.jpg](assets/self_qc/drive_050.jpg) | Rà van, xe bán tải thuộc car, và các xe bị che/cắt khung trong hàng xe. |

Trong lượt CP3 này đã bỏ 2 box ngoài scope và đổi 1 lớp car → van, từ 1.184 xuống 1.182 box. Các thay đổi được đồng bộ vào [CVAT task 5 / job 5](http://localhost:8080/tasks/5/jobs/5) và đọc lại để xác nhận.

Cách kiểm nội dung: lần lượt quét từng làn/hàng xe để tìm thiếu hoặc trùng; kiểm từng xe thuộc đúng một trong bốn lớp; so sánh bốn cạnh box với phần thân xe nhìn thấy; kiểm riêng xe bị che và cắt khung. Checker chỉ kiểm số file, basename, năm cột, ID lớp, miền tọa độ và biên box; không chứng minh đủ xe, đúng lớp hoặc box sát.

## 4. CP4 — Bản nhãn ban đầu

- Commit khóa nhãn ban đầu: [`9420f27238129b4ed2a9851db5fac3d55e98471a`](https://github.com/trxuanbac/K5-L2A-DAY02-TranXuanBac-2A202603132-ObjectDetection/commit/9420f27238129b4ed2a9851db5fac3d55e98471a), `Lock initial AI-assisted train annotations after CP3 QC`.
- Đã tạo fork cá nhân và push commit khóa nhãn lên nhánh `main`. Các commit cập nhật báo cáo về sau giữ nguyên commit nhãn ban đầu này.
- Chưa xác nhận nộp link trên LMS hoặc mốc khóa chung của lớp.
- Gold train được học viên cung cấp và đọc sau khi commit nhãn ban đầu đã được push. Khi rework sẽ tạo commit mới và giữ bản ban đầu, không viết lại lịch sử.

## 5. CP5 — Gold QC và rework

Đã nhận `gold_train_labels.rar` ngày 09/11/2026, giải nén bằng `bsdtar` vào thư mục `gold_train_labels/` được Git bỏ qua. Gold có đủ 50 file, cùng basename ảnh train và đạt checker cấu trúc. SHA-256 archive: `10be4e7fef7181882d30681414564ff975e3c011697e9989ab9c32d52f3f9b3f`.

Đã chạy nguyên bản evaluator, trước mọi rework:

```bash
python3 tools/evaluate_annotations.py \
  --student-labels dataset/labels/train \
  --gold-labels gold_train_labels \
  --output outputs/annotation_evaluation.json
```

| Chỉ số | Lần đầu, gold nguyên trạng | Sau rework |
|---|---:|---:|
| TP | 452 | Chưa chạy |
| FP | 730 | Chưa chạy |
| FN | 201 | Chưa chạy |
| Precision | 0,3824 | Chưa chạy |
| Recall | 0,6922 | Chưa chạy |
| F1 | 0,4926 | Chưa chạy |
| mean_iou_of_matches | 0,8428 | Chưa chạy |
| passed | false | Chưa chạy |

Kết quả lưu tại [outputs/annotation_evaluation.json](../../outputs/annotation_evaluation.json). Matching một-một, cùng ID lớp, IoU ≥ 0,50. Mean IoU đạt 0,70 nhưng F1 chưa đạt 0,80; chưa đủ điều kiện train.

Đã mở đối chiếu ảnh gốc, nhãn hiện có và gold trên `drive_001`, `008`, `018`, `028`; không suy nguyên nhân chỉ từ FP/FN:

- Trong `drive_008`, xe buýt dài có raw gold ID 3, xe tải chở hàng có ID 2, các van có ID 1. Schema của repo là `bus=2`, `truck=1`, `van=3`. Đây là bất nhất cần xác nhận ánh xạ, không phải lý do tự đổi schema của nhãn train.
- Trong `drive_001`, gold chỉ có 5 box trên xe đang lưu thông; hàng xe đỗ bên trái không được gán trong gold. `drive_028` tương tự: gold có 3 box xe lưu thông và bỏ qua hàng xe đỗ hai bên. Guideline hiện tại yêu cầu gán xe đỗ nếu nhận diện được, nên có bất nhất phạm vi.
- Đối chiếu hình học theo box gold: 600/653 box gold có một box học viên chồng lấp IoU ≥ 0,50 khi tạm bỏ điều kiện lớp để chẩn đoán; đây không phải kết quả evaluator chính thức hoặc chỉ số đạt gate.

Học viên đã chọn bảng ID `0=car, 1=truck, 2=bus, 3=van` và xác nhận **vẫn gán xe đỗ như guideline hiện tại**. Lần đánh giá nguyên trạng được giữ để truy vết sự khác biệt ID/phạm vi. Sau đó học viên đã chốt bộ gold làm nguồn nhãn cho lần huấn luyện tham chiếu, dùng raw ID phù hợp với cả gold và eval; không sửa ngược nhãn CP4 hoặc thay điểm CP5.

Giữ nguyên 50 file nhãn học viên của commit CP4; không đổi nhãn xe đỗ thành nền hoặc hoán đổi ID để làm tăng điểm. Gold nguyên trạng được giữ nguyên; ảnh so sánh có gold chỉ lưu local trong `outputs/gold_qc/`, không đưa lên GitHub. Chưa có kết quả sau rework, CP5 của nhãn học viên vẫn chưa đạt.

Sau đó học viên chốt bộ gold làm dữ liệu huấn luyện và yêu cầu chạy train. Đã chuyển sang thí nghiệm riêng **gold_reference**, dùng trực tiếp 653 box gold, không coi đây là rework đã đạt gate của nhãn học viên. Xem [GOLD_TRAINING.md](GOLD_TRAINING.md).

## 6. Huấn luyện và đánh giá — gold_reference

**Đã huấn luyện YOLO26n đủ 50 epoch trực tiếp trên gold đã chốt theo yêu cầu học viên**, sau đó validation riêng bằng `best.pt` và predict đủ 10 ảnh eval. Đây là thí nghiệm tham chiếu theo gold, không phải bằng chứng nhãn học viên đã đạt gate CP5.

- Runtime local: Ultralytics 8.4.208, Torch 2.15.0; 5 epoch đầu trên Apple MPS, resume từ epoch 6 trên CPU khi MPS ngừng cập nhật. Giữ optimizer/EMA; log CSV có đúng 50 dòng epoch 1–50.
- Thông số: YOLO26n pretrained, ảnh 640, batch 8, seed 20260909, AMP tắt, workers 0; optimizer auto chọn AdamW, lr ban đầu 0,00125. Batch cố định và AMP tắt để phù hợp runtime local, khác batch tự động của notebook gốc.
- Train: đúng 50 ảnh, **653 box gold**; validation: đúng 10 ảnh và 113 box. Không đưa eval vào train.
- Raw ID gold/eval: `0=car, 1=van, 2=truck, 3=bus`. Cấu hình runtime `outputs/gold_training/data_gold.yaml` dùng mapping này; `data.yaml` và nhãn CP4 vẫn giữ bản gốc.
- Best checkpoint local: `outputs/gold_training/yolo26_runs/vehicle_gold/weights/best.pt`.
- Chưa có phiên Google Colab đã train. Notebook kết quả đọc artifact của lần train CLI local và hiển thị kết quả thực tế; có thể mở notebook để xem hoặc chạy lại phần đọc artifact.

Kết quả validation của **best checkpoint** (không lấy số của epoch cuối):

| Chỉ số trên 10 ảnh eval | Giá trị |
|---|---:|
| Precision | 0.518141 |
| Recall | 0.646748 |
| mAP50 | 0.645446 |
| mAP50–95 | 0.451602 |

| Lớp theo raw ID gold | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| car | 0.7272 | 0.8936 | 0.9069 | 0.6348 |
| van | 0.2474 | 0.4648 | 0.4828 | 0.3301 |
| truck | 0.4123 | 0.5000 | 0.4950 | 0.3960 |
| bus | 0.6857 | 0.7286 | 0.6971 | 0.4454 |

Bằng chứng:

- [JSON chỉ số model](assets/model_eval_metrics_gold.json), [chẩn đoán prediction](assets/gold_prediction_analysis.json).
- [Log theo epoch](assets/yolo26_runs/vehicle_gold/results.csv), [đồ thị train](assets/yolo26_runs/vehicle_gold/results.png), [confusion matrix](assets/yolo26_runs/vehicle_gold_eval/confusion_matrix_normalized.png).
- [10 ảnh prediction](assets/eval_predictions/visuals/) và [notebook kết quả](../../notebooks/day2_gold_results.ipynb).
- `assets/self_qc/` vẫn là preview nhãn học viên trên train, không phải prediction của model.

Trả lời bốn câu hỏi:

1. **Van có recall thấp nhất (0,4648).** Ở [drive_052.jpg](assets/eval_predictions/visuals/drive_052.jpg), chẩn đoán tại confidence 0,25 và IoU 0,50 ghi nhận 1 van gold, 0 dự đoán van: xe ở đầu làn phía trên bị nhận là car, box chồng lấp IoU 0,9128. Bus hai tầng lớn ở giữa ảnh cũng không có box bus. Các đếm tại một confidence này là minh họa ảnh, không thay cho chỉ số validation quét theo confidence.
2. **mAP50 = 0,6454 và mAP50–95 = 0,4516**, chênh khoảng 0,1938. Cần kiểm độ sát và vị trí bốn cạnh box. Chẩn đoán prediction ở confidence 0,25 có 89 cặp khớp IoU ≥ 0,50, trong đó 10 cặp có IoU < 0,75; một phần box chỉ vượt được ngưỡng lỏng.
3. Có ứng viên nhầm **van → car** trong `drive_052`, `053`, `060` và **truck → van** trong [drive_058.jpg](assets/eval_predictions/visuals/drive_058.jpg) (IoU 0,8781). Đã mở ảnh để đối chiếu. Định nghĩa van/minivan và truck cần nhất quán, nhưng model này train trực tiếp trên gold nên không thể quy lỗi cho guideline nhãn học viên. Train có car 533 box, van 46, truck 20 và bus 54: mất cân bằng lớp là một yếu tố cần xem xét; chưa có thí nghiệm ablation để kết luận nguyên nhân.
4. Giữ 10 ảnh eval tách khỏi train để validation và chọn checkpoint. Vì eval đồng thời dùng chọn best checkpoint, các chỉ số trên không phải kết quả của một test set độc lập.

Gold gốc, nhãn eval và nhãn học viên đều được kiểm tra hash giữ nguyên sau train. Không commit gold train, checkpoint hoặc ảnh train_batch có chứa box gold. Lần train có MPS không bảo đảm deterministic từng bit, dù có đặt seed.

## 7. Khó khăn và tiến độ

- CVAT job 4 chỉ có `drive_002`; đã tạo task 5 chứa đủ 50 ảnh và giữ hai task gốc.
- Repo local ban đầu trỏ tới repo đề bài; đã tạo fork riêng đúng thông tin học viên.
- CP3 đã có checker, mẫu ngẫu nhiên và bằng chứng QC của AI; học viên còn cần tự rà nhãn.
- CP4 đã có SHA và push bản nhãn ban đầu; chưa xác nhận nộp trên LMS hoặc mốc khóa chung của lớp. CP5 nhãn học viên chưa đạt. Theo yêu cầu học viên, đã train mô hình tham chiếu trực tiếp trên gold đủ 50 epoch, có metrics và 10 ảnh prediction; không coi thí nghiệm đó là bằng chứng nhãn học viên đã qua gate.

## 8. Công cụ AI và tài liệu tham khảo

Codex tạo nhãn sơ bộ bằng YOLO11m pretrained trên COCO và chỉnh theo quan sát ảnh; dùng Python/Pillow để render preview, chạy checker và xử lý CVAT/GitHub. Xem [ghi nhận hỗ trợ AI](ANNOTATION_AI.md). Ultralytics 8.4.208 được dùng ở bước tạo nhãn và thí nghiệm YOLO26n theo gold; hai đầu ra được ghi riêng. Lệnh train và quy tắc dữ liệu nằm trong [GOLD_TRAINING.md](GOLD_TRAINING.md).

Đây là nhãn và nội dung báo cáo có AI hỗ trợ. Học viên chưa xác nhận tự kiểm tra, cần ghi đúng phạm vi hỗ trợ và tuân theo [RULES.md](../../RULES.md) trước khi nộp. Không dùng nhãn eval để tạo nhãn train; không tìm nhãn nguồn, tạo gold giả hoặc bịa chỉ số.
