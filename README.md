# Hướng dẫn cài đặt môi trường Conda cho dự án SLRNet

Dự án sử dụng Conda để quản lý môi trường và các gói thư viện. Làm theo hướng dẫn dưới đây để đảm bảo tất cả thành viên
có môi trường giống hệt nhau.

---

## 1. Cài đặt Miniconda (nếu chưa có)

### Trên Windows

1. Tải bản cài đặt Miniconda3 Windows 64-bit tại:  
   [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
2. Chạy file `.exe` vừa tải, chọn **"Just Me"**, tích vào **"Add Miniconda3 to my PATH environment variable"** (rất quan
   trọng).
3. Hoàn tất cài đặt.

### Trên Linux (Ubuntu/Debian)

```bash
# Tải script cài đặt
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Chạy cài đặt
bash Miniconda3-latest-Linux-x86_64.sh

# Nhấn Enter để đọc license, gõ 'yes' đồng ý, và **gõ 'yes' khi hỏi khởi tạo conda**.
# Sau đó đóng terminal và mở lại, hoặc chạy:
source ~/.bashrc