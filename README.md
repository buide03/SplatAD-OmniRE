# SplatAD / OmniRE trên PandaSet

Pipeline nghiên cứu tái tạo nền, xe và người đi bộ trong cùng cảnh. PandaSet cung cấp RGB, LiDAR và cuboid theo UUID. 4D-Humans khởi tạo pose qua pipeline matching và completion của OmniRe. SplatAD giữ renderer RGB, renderer LiDAR và công thức loss để đối chiếu ba cấu hình.

| Cấu hình | Biểu diễn người đi bộ | Tối ưu |
| --- | --- | --- |
| **R0** | Actor rigid trong SplatAD | Gaussian và các nhóm SplatAD |
| **R1** | Một Gaussian trên mỗi vertex SMPL | Gaussian người, pose cố định |
| **R2** | Khởi tạo giống R1 | Gaussian, root translation, root rotation và 23 rotation khớp |
| **R2/vận tốc 0** | Cùng checkpoint R2 | Đặt riêng vận tốc Gaussian người về 0 khi render |

### Bắt đầu trên Colab

1. Đặt `pandaset.zip` trong `MyDrive/SplatAD+OmniRe/`, mở [PandaSet_Colab.ipynb](PandaSet_Colab.ipynb) bằng [Colab](https://colab.research.google.com/) và mount Drive. Để trống `PANDASET_PATH` để ưu tiên ZIP trong thư mục dự án, rồi tự dò My Drive và Shared Drives nếu ZIP này vắng. Có thể chỉ định ZIP khác hoặc thư mục đã giải nén trong `PANDASET_PATH`. Colab giải nén tạm các cảnh đã chọn vào `/content/pandaset-from-drive` và kiểm tra ảnh, scan LiDAR và cuboid của cảnh.
2. Chạy cell **Xem video Gốc và Pose SMPL** để xem cảnh đầu tiên từ Drive. Khi train R1/R2, chọn GPU và tải gói `SMPL_python_v.1.1.0.zip` theo hướng dẫn OmniRe. Đặt file `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` vào `MyDrive/SplatAD+OmniRe/`. Colab cũng nhận cùng bản v1.1 dưới tên `SMPL_NEUTRAL.pkl`. [Trang tải SMPL](https://smpl.is.tue.mpg.de/download.php)
3. Mặc định `STAGE = "curriculum"` train R2 qua `028 → 011 → 008` với 1.200 bước mỗi cảnh. Chọn `STAGE = "scene"` cho một cảnh hoặc sửa `SCENES` để chọn cảnh. Dùng `SCENES = "all"` cùng `VARIANTS = ("R0",)` để chạy bộ dữ liệu theo cấu hình rigid. Các preset `smoke`, `pilot` và `confirm` lần lượt dùng 200, 1.200 và 4.000 bước.
4. Với mỗi cảnh, notebook xuất clip gốc và clip tái tạo `front-triptych.mp4` sau khi có checkpoint. Chọn **Gốc** hoặc **Render** trong dashboard để phát cả ba camera cùng lúc. Trong `MyDrive/SplatAD+OmniRe/`, `inputs/` và `pose-init/` chứa dữ liệu đã chuẩn bị, `runs/` chứa checkpoint, `artifacts/` chứa video, monitor và metric.

### PSNR và SSIM trong lúc train

- **SplatAD gốc.** Notebook chạy mô hình SplatAD tại commit đã pin trước mỗi cấu hình nghiên cứu. Hai bên dùng cùng cảnh, camera, số bước và ngân sách seed phù hợp với Colab. SplatAD dùng thiết lập actor và refinement của mô hình gốc. R0 vẫn là cấu hình đối chứng số Gaussian cố định của repo.
- **Bộ đo cố định.** Mặc định lấy 8 ảnh cách đều cho mỗi camera train. Điểm toàn cảnh được trung bình theo ảnh. Điểm xe và người được trung bình theo crop từ cuboid PandaSet đã chiếu lên ảnh. Crop mở rộng tối thiểu 64 × 64 px, nên có thể chứa nền hoặc vật che. Dashboard đặt kết quả người đi bộ lên đầu bảng, hiển thị số ảnh/crop và dùng cùng bộ đo cho mọi mốc. Mục Pose SMPL bổ sung số đo độ khớp mesh và cập nhật pose cho các track được theo dõi.
- **Hai mốc đối chiếu.** Lưu `initial.ckpt` trước lần cập nhật đầu tiên của run. Ghi `quality.json` ở bước 0, mỗi 250 lần cập nhật (50 với smoke) và bước cuối. Bảng hiển thị giá trị và chênh lệch so với SplatAD ở cùng bước, sau đó so với checkpoint đầu của run đang xem. Số liệu cập nhật khi Colab ghi mốc mới.

Bootstrap pin Python 3.12, PyTorch 2.7.1, CUDA toolkit 12.8/12.6 và các commit trong [versions.json](versions.json). Lần render đầu JIT-build fork GSplat. Đặt `CUDA_HOME` tới toolkit tương ứng nếu máy có nhiều bản CUDA. Giảm `--seed-points`, `--far-points` hoặc train ít camera hơn nếu GPU Colab thiếu VRAM. Dùng cùng các giá trị khi so sánh. [Mã nguồn SplatAD](https://github.com/georghess/neurad-studio/tree/aea389a936890eb48e1b43a493a52c0194559b48)

### Khởi tạo SMPL và lưu Drive

Notebook chạy OmniRe Option 2 để chiếu cuboid người lên ảnh, trích pose bằng 4D-Humans từ sáu camera, ghép tracklet và hoàn thiện các frame còn thiếu. Bản vá trích xuất ghi thêm mặt nạ pose trực tiếp cho monitor. Adapter đọc `smpl.pkl`, đổi tọa độ và gắn UUID để train SplatAD. Kết quả lưu trong `MyDrive/SplatAD+OmniRe/pose-init/omnire-option2/` theo phiên bản và mã băm của SMPL Neutral. [Chuẩn bị pose OmniRe](https://github.com/ziyc/drivestudio/blob/e59bda4fa681f829dbb1d65f0de582b0f633c450/docs/HumanPose.md)

- `OMNIRE_REFERENCE="off"` là mặc định và dùng kết quả trích xuất tại chỗ của OmniRe làm pose khởi tạo.
- `"compare"` ghi chênh lệch tịnh tiến, góc xoay gốc và góc xoay khớp trên các mốc chung với bản OmniRe công bố.
- `"fill"` dành cho lần chạy thử kết hợp hai nguồn. Chế độ này bù mốc còn trống khi bản công bố có pose cùng UUID và frame, rồi căn tâm mesh lại theo cuboid nếu shape khác nhau.
- Drive lưu `smpl.pkl` do OmniRe xuất, `tracks.json` dùng để train, manifest và báo cáo `comparison.json`. Khi bật đối chiếu hoặc kết hợp, Drive lưu thêm JSON của nguồn công bố và báo cáo ghi độ chênh pose trên các mốc chung.

### Quy ước và số liệu

- **Pose.** JSON gắn UUID sau khi đối chiếu ID nội bộ và sibling LiDAR. Rotation root lấy từ camera 4D-Humans thực sự chọn. Translation dịch tâm bbox mesh đã skinning về tâm cuboid. Timestamp tính theo mốc giữa ảnh, xấp xỉ bằng timestamp camera PandaSet trừ 10 ms. Adapter đổi sang hệ cảnh và trừ epoch bằng độ chính xác đầy đủ trước khi cast float32.
- **Gaussian.** DA-pose canonical với trọng số LBS cố định, tâm tại vertex, quaternion theo pháp tuyến, scale theo diện tích và latent 13 kênh. Gaussian SMPL khởi tạo opacity riêng ở 0.99 theo OmniRe. Các lượt Colab có R1/R2 ghi checkpoint và metric dưới mã run mới để đo từ bước 0. Tâm và hướng nhận gradient ở cả hai nhánh renderer. Vận tốc world nội suy từ biến đổi khớp lân cận. Các cấu hình giữ số hạt cố định nên callback refinement tắt. UUID không có track SMPL đủ dùng đi theo actor rigid. Trong khoảng trống dài của track đã chọn, opacity của SMPL bằng 0.
- **Kiểm kê.** Cự ly là khoảng cách từ tâm camera đến tâm cuboid. Nhóm gần ≤15 m, trung bình 15–30 m, xa ≥30 m. Ngưỡng chiều cao bbox mặc định là 28 px. Bộ kiểm kê phân biệt pose trực tiếp từ 4D-Humans với pose do OmniRe nội suy. Ghi rõ trường che khuất và tay/chân nhìn thấy qua review thủ công.
- **RGB.** Render đủ nền, xe và người tại frame gốc. Tính MAE, PSNR, SSIM và LPIPS AlexNet cho toàn ảnh và crop GT. Crop mở rộng tối thiểu 64 × 64 px và giữ mật độ pixel. Trường hợp ảnh khớp hoàn toàn dùng PSNR 120 dB để lưu JSON.
- **LiDAR.** Range MAE tính trên tia có phản hồi GT nằm trong cuboid của UUID, xác định từ chính raster đã dùng để render. Xuất số hit trước và sau lọc quantile. Thiếu hit được ghi là thiếu mẫu. Range panel hiển thị batch raster đầu tiên, metric dùng toàn bộ batch.
- **Tổng hợp.** Đối chiếu trên khóa scene/UUID/camera/frame chung có SMPL hợp lệ, sau đó trung bình quan sát → UUID → scene. LiDAR gộp tia theo scene/UUID/scan trong mỗi bin. `comparison.csv` và `comparison.json` kèm số mẫu và kết quả theo scene.
- **Phát lại.** Mặc định MP4 H.264, CRF 16 và 60 FPS cho bản render. Ba view `front_left`, `front`, `front_right` chia sẻ cùng lưới timestamp từ quan sát đầu tiên đến cuối cùng. Pose cảnh dùng mốc giữa phơi sáng của camera trước làm tham chiếu, còn từng camera giữ hiệu chỉnh rolling shutter. Render lần lượt mỗi view ở độ phân giải gốc để giảm VRAM. Ba clip được ghép trái, giữa, phải vào một MP4 tối đa 3840 px ngang. Dashboard giữ vị trí phát theo cảnh khi đổi giữa Gốc và Render.
- **Chuyển động giữa các mốc.** `REPLAY_INTERPOLATION="smooth"` dùng PCHIP cho vị trí camera và gốc SMPL, cùng spline quay cho hướng camera. Các mốc đầu vào được giữ nguyên. Khoảng trống dài của track SMPL vẫn theo quy tắc hiện diện của mô hình. Bản render truy vấn cảnh ở từng mốc đầu ra. Đổi chế độ nội suy hoặc `REPLAY_FPS` chỉ cần chạy lại bước xuất video nếu đã có checkpoint. Dùng `linear` để đối chiếu với nội suy cũ.

### Thứ tự huấn luyện theo độ khó

| Thứ tự | Cảnh | Quan sát ưu tiên trên ba camera trước |
| --- | --- | --- |
| 1 | `028` | Hai người đi bộ cách 7,9 m và 8,8 m ở `front_left`, frame 64 |
| 2 | `011` | Nhiều người cùng xuất hiện, có silhouette vẽ tay ở `front_right`, frame 8 |
| 3 | `008` | Người cách 30–33 m và bị che chân ở `front`, frame 16 và 46 |

Thứ tự nằm trong [protocol/curriculum.json](protocol/curriculum.json). Colab đọc cảnh tiếp theo từ ZIP sau khi xử lý cảnh trước. Mỗi cảnh có checkpoint riêng và vẫn train RGB/LiDAR toàn cảnh. Các mốc SMPL đã duyệt đi vào monitor, còn quan sát bị che được xem bằng overlay và RGB trên vùng có vật che. Tổng hợp metric chuyển động chi chỉ dùng các quan sát đạt điều kiện hiển thị.

### Cảnh pilot đã khảo sát

| Cảnh | Vai trò | UUID có SMPL nhóm gần | UUID có SMPL nhóm xa |
| --- | --- | ---: | ---: |
| `011` | Gần và xa cùng góc nhìn | 13 | 12 |
| `028` | Hai người đi bộ ở khoảng 8–9 m trong `front_left` | 2 | 2 |
| `008` | Sàng lọc nhiều track ở ≥30 m | 47 | 30 |

Số liệu lấy từ sáu camera, pose OmniRe công khai và SMPL Neutral v1.0 dùng cho phép kiểm kê CPU. [protocol/pilot.json](protocol/pilot.json) ghi số frame, độ bao phủ và hit LiDAR trước review. Các JSON review đi kèm ghi những crop đã kiểm tra. Cảnh 008 có nhiều vùng bị lan can, xe hoặc bồn cây che chân. Đánh giá chuyển động chi ở cự ly xa cần bổ sung review trực tiếp trên frame đã chọn.
### Monitor hiệu chỉnh SMPL

- **Trích xuất pose.** Mục **Pose SMPL** đếm track đã chọn và gộp theo từng cặp người–frame ở ba camera trước. Pose trực tiếp đến từ 4D-Humans. Pose nội suy đến từ bước hoàn thiện pose của OmniRe.
- **Mesh chiếu.** So sánh RGB gốc, mesh khởi tạo và mesh hiệu chỉnh với cùng hiệu chuẩn camera. IoU dùng polygon vẽ tay và lỗi điểm chiếu dùng keypoint có nhãn. Mở **Cập nhật pose trong lúc train** để xem lịch sử, gradient và phép thử cập nhật.
- **RGB vùng người.** Monitor tính L1/MAE từ ảnh toàn cảnh đã render trên cùng mask người hoặc crop bbox cố định. So sánh qua các mốc bằng cột `RGB MAE`. Đây là chỉ số theo dõi vùng người. Loss train tiếp tục lấy từ cấu hình SplatAD toàn cảnh.
- **Probe gradient R2.** Tính gradient từ loss RGB toàn cảnh của cùng một ảnh, thử một bước Adam mới với learning rate hiện hành và chỉ đổi các hàng pose đang hoạt động của UUID này. Monitor đo lại loss, RGB MAE và mesh IoU sau bước thử rồi khôi phục pose. Bảng cập nhật train thật ghi riêng norm gradient và dịch chuyển tham số khi batch train lấy đúng ảnh đó. IoU tăng hoặc lỗi điểm chiếu giảm mới là dấu hiệu mesh sát ảnh hơn.

Cảnh `011/front_right` có một silhouette gần được vẽ tay tại frame 8 trong [protocol/011-review.json](protocol/011-review.json). Với quan sát khác, dùng `pose_monitor: true` trong tệp review để chọn frame theo dõi và `silhouette_xy` để khai báo polygon theo tọa độ pixel ảnh gốc. Dashboard tự đọc `monitor.json` và ảnh theo mốc từ thư mục `artifacts/<run_id>/<scene>-<variant>-pose`. Smoke ghi mốc 0, mỗi 50 bước và bước cuối. Khi resume, dùng đúng checkpoint, cấu hình và manifest đã lưu cho run.

### Nhãn kiểm tra thủ công

Đặt tệp `inputs/011-review.json` trên Drive trước khi tạo manifest. Để tạo lại một bộ probe với nhãn mới, dùng manifest và mã run mới cho cả ba cấu hình.

```json
{
  "observations": [
    {
      "uuid": "UUID từ bộ kiểm kê",
      "camera": "front_right",
      "frame": 30,
      "bbox_xyxy": [812, 442, 860, 570],
      "occlusion": 0.15,
      "limbs_visible": true,
      "keypoints": {
        "left_ankle": {"xy": [835, 548], "vertex_id": 0}
      }
    }
  ],
  "foot_contacts": [
    {
      "uuid": "UUID từ bộ kiểm kê",
      "camera": "front_right",
      "vertex_id": 0,
      "start_frame": 25,
      "end_frame": 30
    }
  ]
}
```

Các tọa độ và `vertex_id` ở trên minh họa định dạng. Chọn ID trên mesh SMPL đã ghép đúng điểm gối, cổ chân, gót hoặc mũi chân. Bỏ qua trường thiếu nhãn. Panel hiển thị GT, overlay mesh khởi tạo, overlay mesh hiện tại, RGB render và sai số ảnh cho cùng crop.

### Chạy dashboard local

```bash
python -m research.dashboard --artifacts /path/to/artifacts --port 8765
```

Mở [http://127.0.0.1:8765](http://127.0.0.1:8765). Bấm tên camera để xem riêng và bấm **Cả ba** để trở về khung đồng bộ. Vị trí phát được giữ khi đổi góc hoặc thẻ. Trong **Pose SMPL**, nút **Tới khung hình** mở đúng camera và mốc quan sát. Các tệp đánh giá RGB/LiDAR lưu cùng artifact để phục vụ đối chứng.

Các lệnh riêng có thể chạy từ root repo sau khi kích hoạt `.deps/train-env/activate-research`.

```bash
python -m research.train --help
python -m research.probe --help
python -m research.evaluate --help
python -m research.replay --help
python -m research.compare --help
python -m pytest -q
```

### Mã nguồn và dữ liệu

- [splatad.patch](splatad.patch) thêm UUID và điểm ghép Gaussian chung trước hai lời gọi GSplat ở commit đã pin.
- [research](research/) chứa export, adapter LBS, train, đo metric, bản phát lại và server dashboard.
- [THIRD_PARTY.md](THIRD_PARTY.md) ghi các phụ thuộc và nguồn mã.
- Chỉ nạp các tệp pose pickle, cấu hình YAML và checkpoint từ nguồn tin cậy. Các tệp này chứa định dạng tuần tự hóa có thể thực thi mã khi nạp.
