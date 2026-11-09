# Báo cáo cá nhân — Trần Xuân Bắc — Ngày 2 Object Detection

## 1. Thông tin học viên

- Họ tên: Trần Xuân Bắc.
- MSSV: 2A202603132.
- Lớp: L2A.
- Email: chưa bổ sung.
- Repo bài làm: https://github.com/trxuanbac/K5-L2A-DAY02-TranXuanBac-2A202603132-ObjectDetection.
- [Mở notebook trên Colab](https://colab.research.google.com/github/trxuanbac/K5-L2A-DAY02-TranXuanBac-2A202603132-ObjectDetection/blob/main/notebooks/day2_annotation_to_yolo.ipynb). Chưa có phiên Colab đã chạy.

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

Học viên đã chọn bảng ID `0=car, 1=truck, 2=bus, 3=van` và xác nhận **vẫn gán xe đỗ như guideline hiện tại**. Theo hai quy tắc này, file gold đang có ví dụ không khớp về ID và phạm vi. Cần giảng viên kiểm tra hoặc cung cấp bộ gold phù hợp trước khi dùng điểm này để kết luận chất lượng nhãn và rework theo gold.

Quyết định hiện tại: giữ nguyên 50 file nhãn train của commit CP4; không đổi nhãn xe đỗ thành nền hoặc hoán đổi ID lớp để làm tăng điểm. Gold nguyên trạng được giữ nguyên; ảnh so sánh có gold chỉ lưu local trong `outputs/gold_qc/`, không đưa lên GitHub. Chưa có kết quả sau rework và chưa đủ điều kiện CP6.

## 6. CP6 — Train và đánh giá YOLO

**Chưa chạy huấn luyện/validation/predict:** CP5 đã chạy nhưng F1 = 0,4926 chưa đạt điều kiện để đi tiếp. Notebook được cấu hình repo cá nhân và kiểm tra gold gate trước khi train.

- Model dự kiến: YOLO26n, 50 epochs, ảnh 640, seed 20260909 theo notebook.
- Split giữ nguyên: 50 ảnh train, 10 ảnh eval làm validation và chọn best checkpoint; eval không phải test độc lập.
- Chưa có runtime Colab, best checkpoint, precision, recall, mAP50, mAP50–95, hoặc 10 ảnh prediction của model đã train.
- Các ảnh ở `assets/self_qc/` là **preview nhãn train**, không phải kết quả dự đoán trên eval.

Phần phân tích:

1. Lớp có recall thấp nhất và ảnh minh họa: chờ kết quả validation thực tế.
2. Nếu mAP50 cao nhưng mAP50–95 thấp: cần kiểm độ bám sát và vị trí bốn cạnh box, không kết luận chỉ từ một chỉ số. Chưa có kết quả của bài này để khẳng định tình huống đó xảy ra.
3. Cặp lớp và lỗi liên quan guideline: van/minivan và car là ca cần kiểm thêm; CP3 đã đổi một xe trong `drive_019`. Chưa có đối chiếu gold hoặc confusion matrix để kết luận lỗi model.
4. Không dùng eval để train vì cần giữ validation riêng để đánh giá và chọn checkpoint. Vì validation đồng thời dùng chọn best checkpoint, kết quả không phải đánh giá trên test độc lập.

## 7. Khó khăn và tiến độ

- CVAT job 4 chỉ có `drive_002`; đã tạo task 5 chứa đủ 50 ảnh và giữ hai task gốc.
- Repo local ban đầu trỏ tới repo đề bài; đã tạo fork riêng đúng thông tin học viên.
- CP3 đã có checker, mẫu ngẫu nhiên và bằng chứng QC của AI; học viên còn cần tự rà nhãn.
- CP4 đã có SHA và push bản nhãn ban đầu; chưa xác nhận nộp trên LMS hoặc mốc khóa chung của lớp. CP5 đã có kết quả ban đầu nhưng chưa đạt và phát hiện bất nhất ID/phạm vi; CP6 chưa chạy. Chưa đủ dữ kiện để tự nhận hoàn thành toàn bộ rubric.

## 8. Công cụ AI và tài liệu tham khảo

Codex tạo nhãn sơ bộ bằng YOLO11m pretrained trên COCO và chỉnh theo quan sát ảnh; dùng Python/Pillow để render preview, chạy checker và xử lý CVAT/GitHub. Xem [ghi nhận hỗ trợ AI](ANNOTATION_AI.md). Ultralytics 8.4.208 được dùng ở bước tạo nhãn; đây không phải kết quả train YOLO26n.

Đây là nhãn và nội dung báo cáo có AI hỗ trợ. Học viên chưa xác nhận tự kiểm tra, cần ghi đúng phạm vi hỗ trợ và tuân theo [RULES.md](../../RULES.md) trước khi nộp. Không dùng nhãn eval để tạo nhãn train; không tìm nhãn nguồn, tạo gold giả hoặc bịa chỉ số.
