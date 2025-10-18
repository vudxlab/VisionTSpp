# VisionTS Local Customisation Notes

Tài liệu này ghi lại toàn bộ thay đổi chúng tôi đã áp dụng lên thư viện VisionTS nội bộ (thư mục `src/uni2ts/model/visionts` và các tiện ích liên quan) nhằm phục vụ bài toán khôi phục dữ liệu GNSS bị thiếu.

---

## 1. Cấu trúc thư viện & import
- Loại bỏ các `sys.path.append("/home/mouxiangchen/VisionTS/visionts")` cứng trong `model.py`, `model_3heads.py`, `model_univariate.py` để đảm bảo thư viện hoạt động độc lập trong workspace hiện tại.
- Sử dụng import tương đối (`from . import models_mae, util`) để dễ dàng phân phối và tái sử dụng.

## 2. Bổ sung tiện ích khôi phục missing data
- `src/uni2ts/model/visionts/imputer.py` (mới): triển khai `VisionTSMissingImputer` cùng `MissingInterval` hỗ trợ chia các đoạn thiếu thành nhiều batch nhỏ (`max_horizon`) và dự báo nối tiếp cho tới khi lấp đầy toàn bộ khoảng trống.
- `src/uni2ts/model/visionts/__init__.py`: export `VisionTSMissingImputer` và `MissingInterval` để notebook/ứng dụng bên ngoài có thể import trực tiếp.

## 3. Điều chỉnh pipeline huấn luyện VisionTS
- `src/uni2ts/model/visionts/pretrain.py`:
  - Thêm tham số `task_mode` (mặc định `forecast`) cho phép chuyển sang chế độ sinh dữ liệu phục vụ imputation (`imagify` sẽ cắt đoạn thiếu ở giữa).
  - Ghi nhận các metadata mới (`impute_context_start`, `impute_missing_start`, `impute_missing_length`) khi đang ở chế độ imputation.
  - Bật/giảm log ảnh trong TensorBoard thông qua tham số cấu hình (`log_image_step`), giữ nguyên cơ chế trực quan hóa sẵn có.

- `cli/conf/pretrain/model/visionts.yaml` & `cli/conf/pretrain/model/visionts_missing.yaml`: cập nhật `task_mode` và thêm biến thể cấu hình dành riêng cho bài toán lấp missing (giới hạn `max_mask_ratio`, `max_dim`, sử dụng đầu ra quantile).

## 4. Chỉnh sửa transformer `ImagifyTS`
- `src/uni2ts/transform/patch.py`:
  - Dịch toàn bộ chú thích và giải thích nội bộ từ tiếng Trung/Anh sang tiếng Việt để tiện bảo trì.
  - Cho phép `ImagifyTS` nhận `task="impute"`: chọn ngẫu nhiên một khoảng trống nằm giữa chuỗi, đảm bảo context nằm phía trước và chunk dự báo ở giữa.
  - Giới hạn kích thước context/pred theo `max_kernel`, điều chỉnh padding, scale_x… để ảnh đầu vào giữ đúng kích thước 224×224.
  - Tăng cường minh họa (màu sắc per-channel) và đảm bảo nvars định màu RGB xen kẽ cho biểu đồ.
  - Khử loại type hint `jaxtyping` khó tương thích; sử dụng `np.ndarray` cơ bản trong `_patchify_arr`.

## 5. Notebook minh họa và quy trình khôi phục GNSS
- `gnss.ipynb`:
  - Đưa `src` vào `sys.path`, đọc dữ liệu `datasets/GNSS.csv`, chuẩn hóa giá trị `-9999`.
  - Gọi `VisionTSMissingImputer` đã tùy biến để khôi phục các khoảng thiếu (tự động chia nhỏ gap nếu dài hơn `max_horizon`).
  - Vẽ biểu đồ trước/sau, hiển thị chi tiết 10 khoảng thiếu dài nhất (cửa sổ = 5 lần độ dài gap, gap nằm giữa) và ghi ra `datasets/GNSS_imputed.csv`.

## 6. Ghi chú vận hành
- `max_horizon` trong imputer mặc định 512 – có thể chỉnh nếu cần dự báo dài hơn trong một lần chạy.
- Cần chắc chắn thư mục checkpoint (`CKPT_DIR`) chứa các file MAE gốc (`mae_visualize_vit_base.pth`, …) trước khi khôi phục dữ liệu.
- Trong huấn luyện, có thể dùng TensorBoard (`tensorboard --logdir outputs`) để xem trực quan quá trình mô hình tái tạo/khôi phục nhờ log ảnh định kỳ.

---

**Tổng quan:** Các thay đổi trên giúp VisionTS++ hỗ trợ trực tiếp bài toán lấp dữ liệu GNSS bị thiếu, có pipeline huấn luyện/imputation rõ ràng, notebook demo chi tiết và tài liệu này đóng vai trò sổ tay bảo trì về sau.
