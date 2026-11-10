# Ghi nhận hỗ trợ AI cho 50 ảnh train

Ngày thực hiện: 09/11/2026. Đây là bộ nhãn được tạo với hỗ trợ AI theo yêu cầu của người dùng, không phải ghi nhận học viên tự gán nhãn. Xác nhận tự rà 10 ảnh CP3 của học viên được bổ sung riêng ngày 10/11/2026.

- Phạm vi: `drive_001.jpg`–`drive_050.jpg`, chỉ split train.
- Công cụ: Codex, Ultralytics 8.4.208 và YOLO11m pretrained trên COCO, chạy local; ảnh đầu vào 1280, confidence 0.18, NMS IoU 0.50.
- Codex quan sát cả 50 ảnh, chỉnh lớp, box trùng, box sai vật thể và một số box bị thiếu/cắt thân xe. Lớp `van` được phân loại lại bằng quan sát vì mô hình COCO không có lớp này.
- Lượt rà lại ảnh có box: `drive_003`, `008`, `012`, `020`, `024`, `026`, `030`, `039`, `042`, `050`. Đây là lượt kiểm tra của AI. Học viên sau đó xác nhận tự rà đủ 10 ảnh CP3 chọn ngẫu nhiên trong `assets/self_qc/`.
- Giai đoạn tạo nhãn ban đầu không dùng eval/gold hoặc truy tìm nhãn nguồn. Gold do học viên cung cấp sau CP4 được dùng cho CP5 và thí nghiệm gold_reference riêng; không dùng eval để sửa nhãn train.

Kết quả ban đầu: 50 file YOLO trong `dataset/labels/train/`, tổng 1.184 box: car 945, truck 31, bus 103, van 105. Sau CP3: 1.182 box (car 942, truck 31, bus 103, van 106); đã bỏ 2 box ngoài scope và đổi 1 car thành van. Xem báo cáo [2A202603132_TranXuanBac.md](2A202603132_TranXuanBac.md) và 10 ảnh chọn ngẫu nhiên trong `assets/self_qc/`. ID lớp giữ đúng `car=0`, `truck=1`, `bus=2`, `van=3`.

CVAT: [task 5 / job 5](http://localhost:8080/tasks/5/jobs/5), chứa đủ 50 ảnh. Đã đọc lại annotations qua API và kiểm tra khớp frame, lớp, tọa độ và số lượng box. Hai task gốc 3 và 4 được giữ nguyên; task mới dùng tên lớp `truck` theo schema, thay cho tên `track` ở hai task cũ. Các box mới mang `source=auto` để ghi nhận AI hỗ trợ.

Lệnh `python3 tools/check_yolo_labels.py` đạt cấu trúc 50/50. Sau rework ngày 10/11/2026: F1 = 0,9508 và mean IoU = 0,8488 theo evaluator cùng lớp ở IoU 0,50, với ánh xạ raw gold sang schema student. Kiểm tra cấu trúc không chứng minh chất lượng annotation.

Bản ZIP YOLO, CVAT XML, dự đoán gốc, các chỉnh sửa và ảnh preview nằm tại `outputs/annotation_assist/` (thư mục này được Git bỏ qua). Token không được ghi vào repo hoặc báo cáo.

Ngày 10/11/2026, Codex rà 50 ảnh để rework theo gold đã chốt: bỏ 576 box ngoài phạm vi/không rõ, đổi 42 lớp, chỉnh 2 box và thêm 1 car. Còn 607 box: car 490, truck 20, bus 53, van 44. Đã đồng bộ lại CVAT và kiểm tra bằng API. Audit và preview mới trong `assets/rework_qc/`. Xác nhận học viên tự rà 10 ảnh áp dụng bản CP3; lượt rework này vẫn ghi nhận do AI hỗ trợ. Gold không bị sửa và không đưa lên GitHub. Bản CP4 được giữ trong lịch sử Git.
