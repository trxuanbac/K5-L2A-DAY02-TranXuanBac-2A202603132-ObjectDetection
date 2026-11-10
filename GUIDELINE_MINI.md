# Mini guideline — bài làm cá nhân

> Điền nội dung này trước khi bắt đầu gán nhãn. Giữ lại các quyết định sau khi self-QC.


****Định nghĩa lớp****

| Lớp | Định nghĩa của bạn | Ví dụ/điểm phân biệt |
| ----- | ----------------------------------------- | --------------------------------------- |
| car | 0-xe con, sedan, SUV, taxi, xe bán tải | xe con, sedan, SUV, taxi, xe bán tải |
| truck | 1-xe tải chở hàng, xe đầu kéo | xe tải chở hàng, xe đầu kéo |
| bus | 2-xe buýt, minibus chở khách | xe buýt, minibus chở khách |
| van | 3-xe van, minivan chở hàng hoặc chở người | xe van, minivan chở hàng hoặc chở người |


## Quy tắc chung

- Box bám sát vật thể nghĩa là: Bao sát toàn bộ phần phương tiện nhìn thấy, hạn chế lấy nền.
- Vật thể bị che khuất: Chỉ gán nhãn phần nhìn thấy, không đoán phần bị che.
- Vật thể bị cắt bởi rìa ảnh: Vẫn gán nhãn nếu nhận diện được, box dừng tại mép ảnh.
- Vật thể rất nhỏ/mơ hồ: Chỉ gán nhãn khi đủ rõ để xác định loại xe, nếu không thì bỏ qua.

## Ghi lại quyết định cho ít nhất ba ca mơ hồ

| Ảnh | Tình huống | Quyết định cuối | Lý do theo schema |
| ------------- | ----------------------------------------------------------------------------------------- | ---------------- | ---------------------------------------------------------------------------------------------------- |
| drive_009.jpg | Một chiếc xe ô tô con bị xe khác che mất phần đầu. | Gán nhãn car (0) | Vẫn nhận diện được đặc điểm thân xe và hình dáng tổng thể. Chỉ vẽ Bounding Box quanh phần nhìn thấy. |
| drive_029.jpg | Một chiếc xe khách bị cắt mất phần đầu do nằm sát cạnh ảnh. | Gán nhãn bus (2) | Có thể xác định rõ đây là xe bus thông qua thân xe và kích thước. Box giới hạn tại cạnh ảnh. |
| drive_001.jpg | Một phương tiện xuất hiện rất nhỏ ở xa, hình ảnh bị mờ, không phân biệt được car hay van. | Không gán nhãn.  | Không đủ đặc điểm trực quan để xác định chính xác một trong bốn lớp đã quy định. |

## Quyết định rework sau khi chốt gold — 10/11/2026

Bản ban đầu được giữ tại commit `9420f27`. Các quyết định dưới đây bổ sung sau khi học viên chốt gold làm tham chiếu; không trình bày như quy tắc đã dùng trước khi nhận gold.

- Giữ ID student `0=car, 1=truck, 2=bus, 3=van`. Gold/eval phát hành dùng `0=car, 1=van, 2=truck, 3=bus`; chỉ chuẩn hóa ID khi so sánh hoặc tạo bản sao runtime, giữ nguyên tọa độ và file nguồn.
- Phạm vi rework theo vùng phương tiện trong gold đã chốt: loại hàng xe đỗ ven đường, làn bên/chiều đối diện và hàng xe quá xa nằm ngoài vùng gán của từng cảnh. Ví dụ `drive_001`, `028` giữ dòng xe trên phần đường chính; `drive_032` giữ phía xe đang tiến tới. Xe đỗ vẫn có loại vật thể; việc bỏ qua ở đây là quyết định phạm vi của bộ gold.
- Gold không kèm polygon ROI hoặc ignore mask; ranh giới phạm vi được ghi theo từng ảnh trong [audit](report/individual/assets/rework_qc/audit.json), chưa thể coi là quy tắc ROI tổng quát cho dữ liệu mới.
- MPV nhỏ có dáng xe con/SUV thuộc car; van thương mại thân hộp, mái cao, khoang chở hàng/người rõ thuộc van. Rà lại các MPV ở `drive_012`, `026`, `050`; không đổi toàn bộ van thành car.
- `drive_019`: xe cứu hộ có thiết bị kéo/chở hàng giữ truck trong vùng gán; không bỏ chỉ vì nó đang đứng yên.
- `drive_009`: box xe bị che bao đúng phần thân thấy được; `drive_050`: thu box xe con về mái xe đó, tránh lấy nóc phương tiện phía sau. `drive_017`: bổ sung car đỏ nhỏ nhìn thấy trên làn chính.

Rework này do Codex hỗ trợ và kiểm tra bằng mắt, có ghi nhận trong báo cáo. Đạt gate không chứng minh đã đủ mọi xe; kết quả còn 54 FN và 8 FP so với gold.
