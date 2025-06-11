ĐÂY LÀ DỰ ÁN HỌC MÁY VÀ ỨNG DỤNG CỦA NHÓM 6 VỚI CHỦ ĐỀ BINH XẬM XÁM
Trong dự án này, chúng tôi đào tạo một mô hình nhận diện lá bài tây với mô hình YOLOv5s. Sau đó thực hiện nhận diện 13 lá bài và hỗ trợ sắp xếp thành bài binh xậm xám

Chỉ cần tập trung vào những điều này:
demo.py
Tệp này là giải thuật xử lý 13 lá bài sắp xếp ra trò chơi binh xậm xám. Điều này phải được chạy đầu tiên.

BXX_Nhom6.pt 
Tệp này để lưu mô hình YOLO tôi đào tạo trên GoogleColab

Thư mục 52cards
Nơi chứa ảnh hiện thị 52 lá bài ra web

Thư mục runs\detect
Nơi tôi lưu trự các giá trị 13 lá dự đoán

Thư mục results
Nơi chứa output của người dùng. Có vai trò sử dùng tiếp tục train cho model

Thư mục test
Nơi chứa input của người dùng

Thư mục card
Các trường hợp ngẫu nhiêu của 13 lá bài

Lệnh thực thi chương trình
1. Chạy ứng dụng demo.py
2. Click vào local URL:  http://127.0.0.1:7860 mở trình duyệt web
3. Chụp ảnh 13 lá bài đưa vào input
4. Submit và đợi kết quả. Kết quả sẽ được lưu vào results

Lưu ý:
Tất cả dữ liệu thử nghiệm trong bài viết này được dùng trong cùng một môi trường đầy đủ ánh sáng.
Môi trường phần cứng sử dụng CPU Intel(R) Core(TM) i5-11400H, RAM 16GB, card đồ họa NVIDIA RTX3050. 
Môi trường hệ thống là Windows 11, Python 3.11.7, torch 2.2.2 , CUDA 12.3