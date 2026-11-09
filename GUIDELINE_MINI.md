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
