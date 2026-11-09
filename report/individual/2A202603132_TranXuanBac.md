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

Ba ca đã ghi trong guideline: xe bị che ở `drive_009`, xe buýt cắt khung ở `drive_029`, phương tiện nhỏ/mờ ở `drive_001`. Phần van/minivan mơ hồ cần học viên đối chiếu thêm với giảng viên; chưa có gold để xác nhận các quyết định lớp.

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

- Commit khóa nhãn ban đầu: sẽ ghi SHA sau khi tạo commit.
- Repo đích đã được tạo bằng fork từ repo đề bài; trạng thái push sẽ cập nhật sau khi kiểm tra GitHub.
- Chưa xác nhận nộp link trên LMS hoặc mốc khóa chung của lớp.
- Chưa đọc hoặc sử dụng gold train. Khi rework sẽ tạo commit mới và giữ bản ban đầu, không viết lại lịch sử.

## 5. CP5 — Gold QC và rework

**Chưa chạy: chưa nhận được đường dẫn gold do giảng viên phát sau mốc khóa.** Chưa có `outputs/annotation_evaluation.json`; không có số liệu TP/FP/FN, precision, recall, F1 hoặc mean IoU để báo cáo.

Sau khi được phát gold đúng mốc, chạy evaluator với matching một-một, cùng lớp, IoU ≥ 0,50. Chỉ đi tiếp khi F1 ≥ 0,80 và mean IoU của các cặp ghép ≥ 0,70. Nếu chưa đạt, kiểm tra ảnh cụ thể, ghi lý do sửa/giữ box và chạy lại; không sửa gold.

## 6. CP6 — Train và đánh giá YOLO

**Chưa chạy huấn luyện/validation/predict:** CP5 chưa đạt điều kiện để đi tiếp. Notebook được cấu hình repo cá nhân và kiểm tra gold gate trước khi train.

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
- CP4 cần lưu SHA/push và xác nhận nộp trên LMS; CP5–CP6 chưa có kết quả. Chưa đủ dữ kiện để tự nhận hoàn thành toàn bộ rubric.

## 8. Công cụ AI và tài liệu tham khảo

Codex tạo nhãn sơ bộ bằng YOLO11m pretrained trên COCO và chỉnh theo quan sát ảnh; dùng Python/Pillow để render preview, chạy checker và xử lý CVAT/GitHub. Xem [ghi nhận hỗ trợ AI](ANNOTATION_AI.md). Ultralytics 8.4.208 được dùng ở bước tạo nhãn; đây không phải kết quả train YOLO26n.

Đây là nhãn và nội dung báo cáo có AI hỗ trợ. Học viên chưa xác nhận tự kiểm tra, cần ghi đúng phạm vi hỗ trợ và tuân theo [RULES.md](../../RULES.md) trước khi nộp. Không dùng nhãn eval để tạo nhãn train; không tìm nhãn nguồn, tạo gold giả hoặc bịa chỉ số.
