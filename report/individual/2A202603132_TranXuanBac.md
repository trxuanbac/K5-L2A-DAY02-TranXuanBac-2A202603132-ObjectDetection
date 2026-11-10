# Báo cáo cá nhân — Trần Xuân Bắc — Ngày 2 Object Detection

## 1. Thông tin học viên

- Họ tên: Trần Xuân Bắc.
- MSSV: 2A202603132.
- Lớp: L2A.
- Email: Bxuan964@gmail.com (theo cấu hình Git local).
- Repo bài làm: https://github.com/trxuanbac/K5-L2A-DAY02-TranXuanBac-2A202603132-ObjectDetection.
- [Notebook Colab học viên đã chạy](https://colab.research.google.com/drive/1YR1tewJQCXHockB095uWxlC7hF9NdBQC#scrollTo=FX-WEVywQc3L). Đã nhận notebook có output và ZIP ngày 10/11/2026, xác nhận train đủ 50 epoch và predict đủ 10 ảnh. Đã phát hiện/sửa lỗi thứ tự ID eval khi đánh giá lại; xem mục 6.
- [Notebook kết quả thí nghiệm gold](../../notebooks/day2_gold_results.ipynb): đọc artifact của phiên train local; không phải phiên train Colab.

## 2. Schema và mini guideline

Áp dụng [GUIDELINE_MINI.md](../../GUIDELINE_MINI.md): `car=0`, `truck=1`, `bus=2`, `van=3`; bao sát phần phương tiện nhìn thấy, giới hạn ở mép ảnh và bỏ vật thể không đủ rõ để xác định lớp.

Ba ca ban đầu: xe bị che ở `drive_009`, xe buýt cắt khung ở `drive_029`, phương tiện nhỏ/mờ ở `drive_001`. Sau khi học viên chốt gold, guideline bổ sung quyết định phạm vi và phân biệt MPV/van trong rework; giữ ID student và ánh xạ tường minh khi so với gold.

## 3. Annotation và CP3 — QC có AI hỗ trợ

- Có 50/50 file YOLO, tương ứng đúng 50 ảnh train.
- Sau CP3: **1.182 box**, gồm car 942, truck 31, bus 103, van 106.
- `python3 tools/check_yolo_labels.py`: **ĐẠT cấu trúc 50/50**.
- Chọn ngẫu nhiên 10 ảnh bằng `random.Random(20261109).sample(sorted(train_images), 10)`; giữ seed để tái lập.
- Codex quan sát cả ảnh có box và ảnh gốc. Ngày 10/11/2026, học viên xác nhận **“Đã tự rà đủ 10 ảnh”** trong `assets/self_qc/`; ghi nhận tại [selection.json](assets/self_qc/selection.json). Xác nhận này áp dụng bản CP3, trước rework CP5.

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

Đã nhận `gold_train_labels.rar` ngày 09/11/2026, giải nén bằng `bsdtar` vào `gold_train_labels/` được Git bỏ qua. Gold có đủ 50 file, 653 box và đạt checker cấu trúc. SHA-256 archive: `10be4e7fef7181882d30681414564ff975e3c011697e9989ab9c32d52f3f9b3f`. Hash 50 file gold giữ nguyên sau rework.

### Chuẩn hóa ID để so sánh cùng lớp

Schema student/CVAT giữ `0=car, 1=truck, 2=bus, 3=van`. Gold phát hành và eval thực tế dùng `0=car, 1=van, 2=truck, 3=bus`, đã đối chiếu ảnh `drive_008`, `052`, `058`. Vì vậy evaluator bổ sung tùy chọn **gold-ID → student-ID** `0:0,1:3,2:1,3:2`, chỉ đổi ID trong bộ nhớ. Box, file gold, thuật toán matching một-một, IoU ≥ 0,50 và hai ngưỡng gate được giữ nguyên. Có test kiểm tra song ánh, giữ tọa độ, matching cùng lớp/một-một và độ phủ tên file.

Lần chạy raw ban đầu giữ tại [annotation_evaluation_initial_raw.json](../../outputs/annotation_evaluation_initial_raw.json). Baseline đã chuẩn hóa ID nhưng chưa sửa nhãn được lưu riêng tại [annotation_evaluation_before_rework.json](../../outputs/annotation_evaluation_before_rework.json). Dùng baseline này để so sánh tác động của rework, tránh gộp cải thiện do ánh xạ ID vào cải thiện nhãn.

| Chỉ số | Raw ban đầu | Trước rework, cùng lớp | Sau rework, cùng lớp |
|---|---:|---:|---:|
| TP | 452 | 557 | 599 |
| FP | 730 | 625 | 8 |
| FN | 201 | 96 | 54 |
| Precision | 0,3824 | 0,4712 | 0,9868 |
| Recall | 0,6922 | 0,8530 | 0,9173 |
| F1 | 0,4926 | 0,6071 | **0,9508** |
| mean_iou_of_matches | 0,8428 | 0,8492 | **0,8488** |
| passed | false | false | **true** |

### Nội dung sửa sau khi chốt gold

Ban đầu học viên chọn giữ xe đỗ theo guideline cũ; sau đó học viên chốt **“bộ gold là chốt”**. Rework ngày 10/11/2026 dùng phạm vi gold làm tham chiếu và cập nhật guideline tương ứng. Bản CP4 vẫn truy lại được tại commit `9420f27`; lịch sử không bị viết lại.

Codex mở rà 50/50 ảnh để xét phạm vi, lớp và cạnh box. Từ 1.182 xuống **607 box**: bỏ 576 box thuộc hàng xe ven lề/làn ngoài phạm vi hoặc xe xa không rõ; đổi 42 lớp sau khi đối chiếu MPV/car/van; chỉnh 2 box theo phần nhìn thấy; thêm 1 car thấy được ở `drive_017`. Nhãn sau sửa có car 490, truck 20, bus 53, van 44. Không sao chép bộ tọa độ gold vào nhãn student.

Ví dụ: `drive_001` và `028` bỏ hàng xe ven đường ngoài phạm vi đã chốt; `drive_009` khôi phục cạnh trên box xe bị che theo phần thân nhìn thấy; `drive_050` thu box xe con về đúng mái xe; `drive_019` vẫn giữ xe cứu hộ trong vùng gán là truck. Quyết định phạm vi không đồng nghĩa mọi xe đỗ đều bị bỏ qua. Các ảnh vẫn còn 54 FN và 8 FP theo evaluator; đạt gate không có nghĩa nhãn hoàn hảo.

Bằng chứng: [audit sửa theo ảnh](assets/rework_qc/audit.json), [10 preview sau rework](assets/rework_qc/). Codex đã mở kiểm tra lại các preview này; xác nhận tự rà 10 ảnh của học viên ở CP3 không được dùng để khẳng định học viên đã rà bản rework.

```bash
python3 tools/check_yolo_labels.py
python3 tools/evaluate_annotations.py \
  --student-labels dataset/labels/train \
  --gold-labels gold_train_labels \
  --gold-class-map 0:0,1:3,2:1,3:2 \
  --output outputs/annotation_evaluation.json
```

Checker **đạt 50/50**. [Kết quả CP5 sau rework](../../outputs/annotation_evaluation.json) đạt cả F1 ≥ 0,80 và mean IoU ≥ 0,70, đủ điều kiện bắt đầu train nhãn student. JSON lưu hash từng file train để notebook từ chối nhãn khác phiên đã chấm. CVAT task 5/job 5 đã đồng bộ và đọc lại đủ 607 box trên 50 ảnh; [xác nhận đồng bộ](assets/rework_qc/cvat_sync_status.json).

Thí nghiệm gold_reference dưới đây đã chạy trước rework, dùng trực tiếp gold theo yêu cầu học viên. Các chỉ số model của nó giữ nguyên và không được đổi tên thành kết quả train nhãn student.

## 6. CP6 — YOLO26n train trên Colab và đánh giá lại đúng ID

### Phiên huấn luyện thực tế

[Colab đã chạy](https://colab.research.google.com/drive/1YR1tewJQCXHockB095uWxlC7hF9NdBQC#scrollTo=FX-WEVywQc3L). Đã nhận `day2_annotation_to_yolo.ipynb` và `day2_colab_outputs.zip` từ học viên ngày 10/11/2026. [Notebook gốc có output](../../notebooks/day2_colab_executed_original.ipynb) được giữ nguyên byte; [manifest nhập kết quả](assets/colab_import_manifest.json) lưu SHA-256 của hai file và checkpoint.

- Notebook log clone commit `8fb51d8`; commit này chứa 50 ảnh train và 607 box student sau rework. Eval vẫn là 10 ảnh riêng, 113 box. Không train trực tiếp bằng gold trong phiên Colab này.
- Train **đủ 50 epoch**, xác nhận bằng đúng 50 dòng `epoch=1..50` trong [results.csv](assets/yolo26_runs/vehicle_annotation/results.csv), sáu cell code đều có output và không có lỗi.
- Runtime Colab: CPU Intel Xeon, Python 3.13.18, Ultralytics 8.4.208, Torch 2.14.1+cpu; không có GPU. YOLO26n pretrained, ảnh 640, seed 20260909, `batch=-1` được AutoBatch chuyển thành 16 trên CPU; optimizer auto chọn AdamW, lr ban đầu 0,00125. Train hoàn tất trong khoảng 0,637 giờ.
- Notebook mở trên Drive là phiên cũ: dù clone repo mới, code notebook không gọi bước chuẩn hóa eval của bản mới. `data.yaml` dùng schema student nhưng eval gốc dùng thứ tự khác; đây là lỗi cấu hình đánh giá, không chứng minh mô hình không nhận được bus.

### Lỗi ID và đánh giá lại checkpoint

Student/model: `0=car, 1=truck, 2=bus, 3=van`. Eval nguồn: `0=car, 1=van, 2=truck, 3=bus`. Log Colab ban đầu đếm “truck=5, bus=2, van=12”, thực chất lần lượt là van, truck, bus. Vì vậy kết luận “bus recall 0%” trong câu trả lời nháp của notebook gốc không được dùng làm kết luận cuối.

Đã dùng `tools/prepare_student_dataset.py` chuẩn hóa ID **trên bản sao** eval bằng `0:0,1:3,2:1,3:2`, giữ nguyên 10 ảnh, tọa độ box và file nguồn. Sau đó chạy `tools/evaluate_colab_checkpoint.py` trên hai checkpoint tải về. Bước train thực hiện ở Colab; bước chấm lại này chạy CPU local, Ultralytics 8.4.208/Torch 2.15.0. Không có phiên train mới và không sửa kết quả notebook gốc.

| Kết quả | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| Colab gốc, best.pt, ID eval chưa khớp | 0,463981 | 0,238666 | 0,276521 | 0,205533 |
| best.pt, chấm lại đúng ID | 0,366920 | 0,365905 | 0,390364 | 0,251436 |
| **last.pt epoch 50, chấm lại đúng ID** | **0,578292** | **0,600400** | **0,566441** | **0,353671** |

Chọn **last.pt** vì có mAP50–95 cao hơn trong hai checkpoint đã lưu. Đây là lựa chọn giữa `best.pt` và `last.pt`, **không chứng minh là checkpoint tốt nhất của toàn bộ 50 epoch**: các checkpoint trung gian không được lưu, còn `best.pt` ban đầu được chọn bằng validation sai thứ tự ID. Muốn chọn best xuyên suốt 50 epoch với eval đúng ID, chạy lại từ đầu notebook chính đã sửa. Chênh lệch giữa hàng đầu và hàng cuối gồm cả sửa ID lẫn đổi checkpoint; không quy hết cho một nguyên nhân.

SHA-256 checkpoint được chọn: `205846fb44de0e33f8bafe2863108341a357e7e3301e4e686d0b6b09fa99cac5`. File `.pt` giữ local, không đưa lên GitHub.

### Kết quả cuối cho checkpoint được chọn

| Lớp student | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| car | 0.8672 | 0.8404 | 0.8911 | 0.5683 |
| truck | 0.2741 | 0.4112 | 0.1928 | 0.1349 |
| bus | 0.8506 | 0.7500 | 0.8332 | 0.4466 |
| van | 0.3213 | 0.4000 | 0.3486 | 0.2648 |

Bằng chứng:

- [JSON chỉ số đã chuẩn hóa](assets/model_eval_metrics.json), [JSON chỉ số Colab gốc](assets/model_eval_metrics_colab_original.json), [chẩn đoán prediction](assets/student_prediction_analysis.json).
- [Đồ thị train Colab](assets/yolo26_runs/vehicle_annotation/results.png): các đường validation theo epoch dùng ID cũ; xem như lịch sử chạy, không coi là đồ thị validation đã sửa.
- [Confusion matrix sau chấm lại](assets/yolo26_runs/vehicle_eval/confusion_matrix_normalized.png), [PR curve sau chấm lại](assets/yolo26_runs/vehicle_eval/BoxPR_curve.png).
- [10 prediction checkpoint được chọn](assets/eval_predictions/visuals/), [10 prediction Colab gốc](assets/eval_predictions/colab_original/visuals/), [notebook xem kết quả và đánh giá lại](../../notebooks/day2_colab_results.ipynb).

Codex đã mở kiểm tra cả 10 prediction sau chấm lại cùng confusion matrix; đây là lượt kiểm tra có AI hỗ trợ. Confusion matrix validation dùng ngưỡng riêng của validator, không đồng nhất với các đếm minh họa tại confidence 0,25 dưới đây.

### Trả lời bốn câu hỏi

1. **Lớp nào có recall thấp nhất? Chỉ ra một ảnh eval minh họa.** Với checkpoint `last.pt` đã chấm đúng schema, **van thấp nhất, recall = 0,4000**; truck = 0,4112, bus = 0,7500 và car = 0,8404. Trong [drive_053.jpg](assets/eval_predictions/visuals/drive_053.jpg), van trắng ở phần đường bên trái bị nhận thành car; box dự đoán khớp box eval với IoU **0,8608**. Ở confidence 0,25, ảnh có 1 van theo eval, 0 dự đoán van, 1 FN van. Đây là minh họa ở một confidence, khác recall validation được tổng hợp theo đường cong.
2. **Nếu mAP50 cao nhưng mAP50–95 thấp, box đang gặp vấn đề gì?** Phiên chấm lại có **mAP50 = 0,5664**, **mAP50–95 = 0,3537**, chênh **0,2128**. Khi tăng yêu cầu IoU, box lệch hoặc lấy thừa/thiếu thân xe mất điểm. Cần kiểm bốn cạnh box, nhất là xe bị che, xe nhỏ và cắt khung. Chẩn đoán ở confidence 0,25 có **93** cặp cùng lớp khớp IoU ≥ 0,50, trong đó **20** cặp có IoU < 0,75. Chênh lệch mAP gợi ý chất lượng định vị cần cải thiện; tự nó không chứng minh annotation train sai.
3. **Có lỗi truck/bus/van/car nào liên quan tới guideline không?** Đã thấy **van → car** trên `drive_052` (IoU 0,8866), `drive_053` (0,8608), `drive_056` (0,8517), `drive_060` (0,8230). Các ca có box khá sát nhưng sai lớp, nên cần rà tính nhất quán khi phân biệt MPV nhỏ với van thân hộp theo guideline. Train có car 490, truck 20, bus 53, van 44, là dữ liệu ít và mất cân bằng; chưa có thí nghiệm tách nguyên nhân để kết luận lỗi chỉ do guideline. Lỗi thứ tự ID eval là lỗi pipeline riêng, giải thích vì sao bảng Colab cũ hiển thị bus recall 0%; không dùng bảng đó để suy ra model không nhận được bus. Không sửa gold/eval để làm tăng điểm.
4. **Vì sao không dùng 10 ảnh eval để train dù chúng có nhãn?** Giữ chúng tách khỏi train để đo khả năng khái quát trên ảnh model chưa học trực tiếp. Train rồi chấm trên chính các ảnh đó gây rò rỉ dữ liệu và làm điểm quá lạc quan. Eval còn được dùng làm validation/chọn checkpoint, kể cả việc so hai checkpoint đã lưu ở đây, nên **không phải test set độc lập**. Giữ nguyên 50 train/10 eval.

## 6b. Thí nghiệm bổ sung — gold_reference local

**Đã huấn luyện YOLO26n đủ 50 epoch trực tiếp trên gold đã chốt theo yêu cầu học viên**, sau đó validation riêng bằng `best.pt` và predict đủ 10 ảnh eval. Đây là thí nghiệm tham chiếu theo gold, không phải bằng chứng nhãn học viên đã đạt gate CP5.

- Runtime local: Ultralytics 8.4.208, Torch 2.15.0; 5 epoch đầu trên Apple MPS, resume từ epoch 6 trên CPU khi MPS ngừng cập nhật. Giữ optimizer/EMA; log CSV có đúng 50 dòng epoch 1–50.
- Thông số: YOLO26n pretrained, ảnh 640, batch 8, seed 20260909, AMP tắt, workers 0; optimizer auto chọn AdamW, lr ban đầu 0,00125. Batch cố định và AMP tắt để phù hợp runtime local, khác batch tự động của notebook gốc.
- Train: đúng 50 ảnh, **653 box gold**; validation: đúng 10 ảnh và 113 box. Không đưa eval vào train.
- Raw ID gold/eval: `0=car, 1=van, 2=truck, 3=bus`. Cấu hình runtime `outputs/gold_training/data_gold.yaml` dùng mapping này; `data.yaml` và nhãn CP4 vẫn giữ bản gốc.
- Best checkpoint local: `outputs/gold_training/yolo26_runs/vehicle_gold/weights/best.pt`.
- Notebook kết quả gold_reference đọc artifact của lần train CLI local. Học viên sau đó cung cấp phiên Colab riêng ở mục 6; hai thí nghiệm có nguồn nhãn và chỉ số riêng.

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
- [10 ảnh prediction](assets/eval_predictions/gold_reference/visuals/) và [notebook kết quả](../../notebooks/day2_gold_results.ipynb).
- `assets/self_qc/` vẫn là preview nhãn học viên trên train, không phải prediction của model.

Trả lời bốn câu hỏi:

1. **Van có recall thấp nhất (0,4648).** Ở [drive_052.jpg](assets/eval_predictions/gold_reference/visuals/drive_052.jpg), chẩn đoán tại confidence 0,25 và IoU 0,50 ghi nhận 1 van gold, 0 dự đoán van: xe ở đầu làn phía trên bị nhận là car, box chồng lấp IoU 0,9128. Bus hai tầng lớn ở giữa ảnh cũng không có box bus. Các đếm tại một confidence này là minh họa ảnh, không thay cho chỉ số validation quét theo confidence.
2. **mAP50 = 0,6454 và mAP50–95 = 0,4516**, chênh khoảng 0,1938. Cần kiểm độ sát và vị trí bốn cạnh box. Chẩn đoán prediction ở confidence 0,25 có 89 cặp khớp IoU ≥ 0,50, trong đó 10 cặp có IoU < 0,75; một phần box chỉ vượt được ngưỡng lỏng.
3. Có ứng viên nhầm **van → car** trong `drive_052`, `053`, `060` và **truck → van** trong [drive_058.jpg](assets/eval_predictions/gold_reference/visuals/drive_058.jpg) (IoU 0,8781). Đã mở ảnh để đối chiếu. Định nghĩa van/minivan và truck cần nhất quán, nhưng model này train trực tiếp trên gold nên không thể quy lỗi cho guideline nhãn học viên. Train có car 533 box, van 46, truck 20 và bus 54: mất cân bằng lớp là một yếu tố cần xem xét; chưa có thí nghiệm ablation để kết luận nguyên nhân.
4. Giữ 10 ảnh eval tách khỏi train để validation và chọn checkpoint. Vì eval đồng thời dùng chọn best checkpoint, các chỉ số trên không phải kết quả của một test set độc lập.

Tại thời điểm train gold_reference, gold gốc, nhãn eval và nhãn học viên được kiểm tra hash giữ nguyên sau train. Nhãn student được rework sau đó, có audit riêng ở CP5. Không commit gold train, checkpoint hoặc ảnh train_batch có chứa box gold. Lần train có MPS không bảo đảm deterministic từng bit, dù có đặt seed.

## 7. Tiến độ và việc còn cần thực hiện

- CP3: checker đạt; học viên xác nhận tự rà 10 preview CP3 ngày 10/11/2026.
- CP4: giữ bản đầu `9420f27` trong lịch sử Git; nhãn và báo cáo đã có trên GitHub. Chưa xác nhận mốc khóa chung của lớp.
- CP5: rework 50 ảnh đạt F1 0,9508 và mean IoU 0,8488; nhãn và CVAT khớp 607 box.
- CP6: đã nhận notebook Colab có output, log đủ 50 epoch, checkpoint và 10 prediction. Đã sửa lỗi ID khi chấm lại hai checkpoint đã lưu; `last.pt` epoch 50 được chọn, có metrics, ảnh minh họa và trả lời đủ bốn câu hỏi. Giới hạn lựa chọn checkpoint được ghi ở mục 6.
- Gold_reference là thí nghiệm local bổ sung, giữ riêng bộ 653 box gold và các chỉ số của nó; không gộp với phiên Colab student.
- **Còn cấp quyền xem notebook cho người chấm và nộp link repo lên LMS bằng tài khoản cá nhân.** Đã có bản notebook gốc tải về để xem output trong repo; chưa xác nhận quyền chia sẻ của link Drive hoặc trạng thái nộp LMS. Không upload gold/checkpoint lên GitHub.

## 8. Công cụ AI và tài liệu tham khảo

Codex tạo nhãn sơ bộ bằng YOLO11m pretrained trên COCO và chỉnh theo quan sát ảnh; dùng Python/Pillow để render preview, chạy checker và xử lý CVAT/GitHub. Xem [ghi nhận hỗ trợ AI](ANNOTATION_AI.md). Ultralytics 8.4.208 được dùng ở bước tạo nhãn, train YOLO26n theo gold và phiên Colab student. Codex kiểm tra artifact Colab, sửa ánh xạ eval trên bản sao, đánh giá hai checkpoint và hỗ trợ viết câu trả lời; các đầu ra và môi trường chạy được ghi riêng. Lệnh train và quy tắc dữ liệu nằm trong [GOLD_TRAINING.md](GOLD_TRAINING.md).

Đây là nhãn và nội dung báo cáo có AI hỗ trợ. Học viên đã xác nhận tự rà 10 ảnh CP3; phần rework 50 ảnh và kiểm tra preview sau sửa do Codex thực hiện. Cần ghi đúng phạm vi hỗ trợ và tuân theo [RULES.md](../../RULES.md) trước khi nộp. Không dùng nhãn eval để tạo nhãn train; không tìm nhãn nguồn, tạo gold giả hoặc bịa chỉ số.
