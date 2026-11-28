# SplatAD / OmniRE trên PandaSet

Pipeline nghiên cứu tái tạo nền, xe và người đi bộ trong cùng cảnh. PandaSet cung cấp RGB, LiDAR và cuboid theo UUID. 4D-Humans khởi tạo pose qua pipeline matching và completion của OmniRe. SplatAD giữ renderer RGB, renderer LiDAR và công thức loss để đối chiếu ba cấu hình.

| Cấu hình | Biểu diễn người đi bộ | Tối ưu |
| --- | --- | --- |
| **SplatAD** | Người đi bộ không được tách khỏi nền | Mô hình gốc tại commit đã pin, MCMC đến 5 triệu Gaussian |
| **R2** | Một Gaussian trên mỗi vertex SMPL. Người không có SMPL là actor rigid | Gaussian, root translation, root rotation và 23 rotation khớp. Nền và xe giữ MCMC và lật actor của SplatAD |
| **R2/vận tốc 0** | Cùng checkpoint R2 | Đặt riêng vận tốc Gaussian người về 0 khi render |
| **R3** | Như R2. Người đi bộ ở xa không có pose 4D-Humans dùng SMPL với chu kỳ bước mượn từ người gần | R2 cộng regularizer SMPL của OmniRe, tiếp đất theo LiDAR, SMPL gắn vào hộp đang học. Chạy thử 10.000 bước |
| **OmniRe** | Mã DriveStudio gốc: SMPL cho người có pose, Gaussian biến dạng cho người còn lại | `configs/omnire.yaml` với `pandaset/6cams`, chỉ thay các thiết lập chung bên dưới |

R2 và R3 chỉ cải tiến phần người đi bộ của SplatAD. Nền, xe, decoder, loss RGB/LiDAR và GSplat để render giữ nguyên của SplatAD. Gaussian người đi qua adapter `scene_gaussians` vào cùng hai lời gọi rasterization RGB và LiDAR.

### Thiết lập chung của bốn mô hình

| Mục | Giá trị |
| --- | --- |
| Cảnh và frame | `030`, toàn bộ 80 frame dùng để train |
| Camera | Sáu camera ở độ phân giải gốc. Camera sau bỏ 260 hàng dưới theo SplatAD khi đo |
| LiDAR | Pandar64 360°. Rasterizer LiDAR của SplatAD chỉ hỗ trợ cảm biến này, nên OmniRe đặt `only_use_360_lidar=True` |
| Ngân sách | 10.000 bước, seed 42. Mọi mô hình dùng chung lịch learning rate (`--schedule-steps` 30.001), nên đặt `R3_STEPS = 30_000` trong notebook sẽ train tiếp R3 từ checkpoint 10.000 |
| Pose người | Cùng `smpl.pkl` từ pipeline 4D-Humans của OmniRe trên sáu camera |
| Đo RGB | Cùng bộ ảnh đo, cùng crop và cùng SSIM `pytorch_msssim` (`research.quality.score_images`) |

Mỗi mô hình giữ lịch train riêng của nó, ví dụ OmniRe tăng dần độ phân giải trong 500 bước đầu. Ngoại lệ duy nhất: OmniRe ngừng tách/nhân Gaussian ở bước 7.000 (`STOP_SPLIT_AT`) thay vì 15.000. Cấu hình gốc load PandaSet ở nửa độ phân giải; ở độ phân giải gốc, lượt 030 có 4,48 triệu Gaussian nền ở bước 7.500 và hết 22 GB GPU của L4. Việc tỉa Gaussian vẫn chạy như cũ. OmniRe không render LiDAR nên chỉ có số đo RGB. Bảng `artifacts/<run_id>/<scene>-summary/summary.md` so bốn mô hình ở các bước 2.500, 5.000, 7.500 và 10.000.

### R3: sửa các vấn đề của R2

Lượt R2 cũ cảnh `030` trên Drive (10.000 bước, khi R2 còn tắt MCMC) cho PSNR vùng người cao hơn SplatAD khoảng 1,5 dB. PSNR toàn cảnh thấp hơn 2,4 dB, vùng xe thấp hơn 1,4 dB, và khoảng cách này tăng dần từ bước 2.000. R3 xử lý từng nguyên nhân:

| Vấn đề của R2 | Nguyên nhân | R3 |
| --- | --- | --- |
| Metric toàn cảnh và xe thấp hơn SplatAD | R2 tắt MCMC (410 nghìn Gaussian nền cố định so với giới hạn 5 triệu) và đặt xác suất lật actor về 0 | Bật lại MCMC và lật actor của SplatAD cho cả R2 và R3, nên ba mô hình chỉ khác cách khởi tạo và chuyển động của người đi bộ. MCMC của gsplat chỉ sửa `gauss_params`, nên Gaussian SMPL không bị ảnh hưởng. Số liệu R2 cũ không so được với R2 mới, cần train lại |
| Người ở xa trượt như một khối | Người không có SMPL đi theo actor rigid | Người đi bộ di chuyển (tổng quãng đường cuboid ≥ 0,5 m) không có pose 4D-Humans dùng được nhận khung xương SMPL gắn vào hộp, với pose khởi tạo từ một chu kỳ bước của người gần (`research/gait.py`), rồi tinh chỉnh pose như mọi track SMPL. Màu Gaussian lấy từ điểm dữ liệu gần nhất trong hộp: mọi điểm Pandar64 trong cuboid của cả chuỗi, tô màu từ camera nhìn thấy nó, cộng điểm lấp trong hình trụ thân người tới 2.000 điểm, chỉ giữ điểm lấp mà ít nhất 2 view thấy đúng người đó |
| Chuyển động người gần giật | Pose từng frame được tối ưu độc lập | `trans_temporal_smooth` (bước nhảy ngẫu nhiên 1–5 pose, w 0,01) và `joint_smooth` (w 0,01) của OmniRe |
| Chân lẫn với mặt đường | Gaussian SMPL lớn hoặc dẹt lan xuống đường, chân không có ràng buộc tiếp đất | `max_s_square` (w 0,05), `sharp_shape` (tỉ lệ ≤ 10), `knn_reg` (w 0,001) của OmniRe. Mặt đường cục bộ fit từ LiDAR: bàn chân đang chạm đất bám mặt đường, không xuyên xuống, không trượt |

Ba thay đổi khác ở adapter SMPL của R3:

| Vấn đề còn lại của R2 | R3 |
| --- | --- |
| Gốc SMPL tách khỏi quỹ đạo cuboid đang học | Gốc SMPL được biểu diễn trong hệ của hộp (`smpl_follow_box`). Mesh = `box2world(t)` ∘ pose cục bộ, nên khi quỹ đạo hộp được tối ưu thì mesh đi theo, và gradient RGB/LiDAR của Gaussian SMPL cũng chỉnh quỹ đạo hộp. Lúc gắn, mesh giữ nguyên vị trí cũ |
| Vận tốc SMPL tính trong `no_grad` | Vận tốc là sai phân trung tâm của Gaussian đã đặt pose tại t ± 10 ms, có gradient, gồm cả chuyển động khớp, gốc và hộp |
| Người xa có SMPL kém vẫn đi theo SMPL | Track chỉ giữ pose 4D-Humans khi có ít nhất `smpl_min_frames` = 5 frame pose trực tiếp, ngoài nhóm xa và bbox cao ít nhất `smpl_min_box_px` = 80 px. Track không đạt mà có di chuyển dùng chu kỳ bước mượn. Ở cảnh 030 cả 8 track SMPL đều đạt (27–77 frame) |

#### Người ở xa: chu kỳ bước mượn thay cho MLP biến dạng

Bản R3 trước dùng `DeformableNodes` của OmniRe (MLP theo vị trí, thời gian và embedding) cho 9 người xa của cảnh 030 (35–66 m, cao 29–96 px). Kiểm chứng trên checkpoint 10.000 bước cho thấy MLP chỉ dịch chân 1–2,4 cm theo thời gian và tắt MLP đổi PSNR vùng người trong khoảng ±0,8 dB: người xa trượt theo hộp. Loss so màu từng pixel không kéo được chân đi vài chục pixel tới vị trí thật, nên MLP không có tiền nghiệm chuyển động thì dừng ở hình trung bình. OmniRe cũng không có cơ chế riêng cho trường hợp này.

R3 hiện tại thay MLP bằng khung xương có dáng đi chuẩn, rồi buộc nó vào bằng chứng của từng khung:

1. **Dáng đi chuẩn của cảnh** (`research/gait.py`). Mỗi người gần (track SMPL gắn hộp) cho một chu kỳ chân lặp lại rõ nhất: độ trễ 0,6–1,6 s cho góc hông và gối khớp nhau ở mọi điểm bắt đầu, chân vung ít nhất 5°. Các chu kỳ được xoay cho cùng đi theo +x, lấy mẫu lại 24 pha, căn pha theo chu kỳ sạch nhất rồi lấy trung bình.
2. **Gắn vào hộp.** Người xa nhận SMPL dáng trung bình gắn vào hộp của nó. Pha tăng theo quãng đường hộp đi được, mỗi chu kỳ một sải bước (sải bước trên mỗi mét chiều cao hộp lấy từ người gần, nhịp bước giữ trong 0,5–1,5 lần nhịp của họ), nên hộp dừng thì chân dừng. Chu kỳ xoay theo hướng đi của hộp, khung chậu nằm giữa hộp, bàn chân trụ chạm đáy hộp. Biên độ vung hông được nhân hệ số 0,5–4 (góc lệch tối đa 35°), chọn sao cho bàn chân trụ đứng yên ở đúng tốc độ hộp: 4D-Humans và việc lấy trung bình nhiều người gần làm biên độ nhỏ đi, ở cảnh 030 bàn chân chỉ vung 0,12–0,22 m trong khi mỗi bước dài khoảng 0,7 m, nên chân trượt theo hộp. `scene_routes.json` ghi `gait_hip_scale`, `foot_slip_before_mps` và `foot_slip_mps`. Người xa chậm hơn 0,2 m/s (`gait_min_speed`, thường là người đứng có nhãn hộp rung) giữ dạng khối cứng như R2. Người xa cao bằng hộp của nó: hệ số dáng SMPL thứ nhất (`betas[0]`, chủ yếu đổi chiều cao) được chọn cho chiều cao mesh bằng chiều cao hộp, giới hạn 1,45–1,95 m và ±3 độ lệch chuẩn; `scene_routes.json` ghi `box_height_m` và `shape_beta0`.
3. **Keypoint 2D** (`research/keypoints.py`). ViTPose (`usyd-community/vitpose-base-simple`, `transformers` 4.53.2) chạy top-down trên bbox của mọi quan sát người đi bộ trong manifest, nới 15% mỗi cạnh, cho 17 keypoint COCO kèm độ tin cậy và camera PandaSet của ảnh đó.
4. **LiDAR từng khung.** File của `research.deform_seed` (v2) giữ các điểm Pandar64 của từng khung trong cuboid của khung đó, bỏ điểm sát đáy hộp (mặt đường).
   Cảnh không có người gần nào đi bộ rõ thì mượn dáng đi chuẩn đã lưu của các cảnh khác: mỗi lượt R3 có người gần ghi dáng đi của cảnh vào `MyDrive/SplatAD+OmniRe/gait-library/<cảnh>.json` (`--gait-library`), cảnh thiếu người gần lấy trung bình các file đó. `scene_routes.json` ghi `gait_source` là `scene` hay `library`. Chưa có thư viện thì người xa giữ actor rigid như R2.
5. **Chọn pha ban đầu** (`research/far_fit.py`). Thử 24 pha, giữ pha cho tổng hai loss dưới đây nhỏ nhất, nên chân nào bước trước khớp với dữ liệu.
6. **Phạt khi train.** Pose train như mọi track SMPL của R3 (tinh chỉnh từng frame, hai loss làm mượt, `local_offset`, quỹ đạo hộp), cộng:
   - `r3_keypoint`: 12 khớp SMPL (vai, khuỷu, cổ tay, hông, gối, cổ chân) chiếu bằng camera của từng quan sát, so với keypoint có độ tin cậy ≥ 0,3, sai số tính theo chiều cao bbox;
   - `r3_lidar_fit`: khoảng cách từ mỗi điểm LiDAR của 4 khung ngẫu nhiên tới tâm Gaussian gần nhất của mesh đã đặt pose, trong hệ cảnh, nên kéo được cả pose lẫn hộp.

   - tiếp đất (`r3_foot_contact`, `r3_ground_penetration`, `r3_foot_slip`): mặt đáy cuboid đã gán nhãn ở từng khung là mặt đường của người xa (LiDAR ở 35–66 m quá thưa để fit mặt đường cục bộ như người gần). Bàn chân trụ bám mặt đáy, không xuyên xuống, không trượt, nên mesh không lơ lửng. Khung bàn chân trụ được nhận ra từ dáng đi lúc khởi tạo (chân cách mặt đáy ≤ 20 cm, chậm hơn 0,65 m/s).

   Cả hai loss keypoint và LiDAR dùng phạt Geman–McClure r²/(r²+σ²) (σ = 0,1 chiều cao bbox và 0,1 m): bậc hai khi gần đúng, bị chặn khi gặp keypoint trái/phải bị đổi hay điểm của vật che. Khác loss so màu, gradient của hai loss này luôn chỉ về đáp án kể cả khi chân đang lệch nhiều pixel.

`scene_routes.json` ghi `smpl_gait` cho người xa, kèm các người cho mượn (`gait_donors`), chu kỳ, pha đã chọn, sải bước, nhịp bước, góc xoay, số quan sát keypoint, số khung LiDAR, sai số keypoint (px) và loss LiDAR lúc khởi tạo, và tỉ lệ vertex tô được màu từ dữ liệu.

Hai loss làm mượt tính trên vị trí khớp trong hệ cảnh (hộp ∘ pose cục bộ), nên phạt cả rung của hộp lẫn rung của pose. Loss `local_offset` (w 0,01) neo gốc cục bộ quanh vị trí ban đầu trong hộp, để hộp gánh phần di chuyển và hai bậc tự do không bù trừ nhau. Monitor ghi thêm độ lệch của hộp đã học so với nhãn PandaSet. Dùng `--no-smpl-follow-box` để train một bản R3 không gắn hộp làm đối chứng.

| Loss R3 | Trọng số | Nhận gradient |
| --- | ---: | --- |
| `r3_trans_smooth` (khung chậu, bước 1–5 pose) | 0,01 | pose SMPL, quỹ đạo hộp |
| `r3_joint_smooth` (24 khớp, 3 pose liên tiếp) | 0,01 | pose SMPL, quỹ đạo hộp |
| `r3_local_offset` | 0,01 | pose SMPL |
| `r3_foot_contact`, `r3_ground_penetration` (người gần: mặt đường LiDAR; người xa: đáy cuboid) | 0,1 | pose SMPL, quỹ đạo hộp |
| `r3_foot_slip` | 0,1 | pose SMPL, quỹ đạo hộp |
| `r3_knn` | 0,001 | Gaussian SMPL |
| `r3_max_scale` | 0,05 | Gaussian SMPL |
| `r3_sharp_shape` (mỗi 10 bước) | 1 | Gaussian SMPL |
| `r3_keypoint` (người xa) | 0,05 | pose người xa, quỹ đạo hộp |
| `r3_lidar_fit` (người xa, 4 khung mỗi bước) | 0,05 | pose và Gaussian người xa, quỹ đạo hộp |

Loss RGB và LiDAR của SplatAD đi tới mọi nhóm: Gaussian nền và xe, Gaussian và pose SMPL của người gần lẫn người xa, quỹ đạo hộp. Tiếp đất áp cho người gần ở frame có mặt đường LiDAR và cho người xa trên đáy cuboid. `tests/test_r3.py::test_r3_gradients_reach_meshes_poses_and_box` kiểm tra ma trận này.

Trọng số theo [configs/omnire.yaml](https://github.com/ziyc/drivestudio/blob/e59bda4fa681f829dbb1d65f0de582b0f633c450/configs/omnire.yaml). OmniRe để `joint_smooth` ở dạng chú thích. R3 bật nó với trọng số đã ghi trong file đó vì R2 bị giật. Với CLI, `--ground` nhận file của `python -m research.ground` `--deform-seed` nhận file `.npz` của `python -m research.deform_seed` (màu và LiDAR từng khung cho người xa) và `--keypoints` nhận file của `python -m research.keypoints`. Trọng số hai loss mới là giá trị khởi đầu, chưa được dò. Người xa có dưới 64 điểm trong file giữ màu xám mặc định của SMPL. Dashboard và `training.jsonl` ghi các loss `r3_*`. `scene_routes.json` ghi UUID nào dùng `smpl`, `smpl_gait` hay `rigid`.

### Bắt đầu trên Colab

1. Đặt `pandaset.zip` trong `MyDrive/SplatAD+OmniRe/`, mở [PandaSet_Colab.ipynb](PandaSet_Colab.ipynb) bằng [Colab](https://colab.research.google.com/) và mount Drive. Để trống `PANDASET_PATH` để ưu tiên ZIP trong thư mục dự án, rồi tự dò My Drive và Shared Drives nếu ZIP này vắng. Có thể chỉ định ZIP khác hoặc thư mục đã giải nén trong `PANDASET_PATH`. Colab giải nén tạm các cảnh đã chọn vào `/content/pandaset-from-drive` và kiểm tra ảnh, scan LiDAR và cuboid của cảnh.
2. Chạy cell **Xem video Gốc và Pose SMPL** để xem cảnh đầu tiên từ Drive. Khi train R2/R3, chọn GPU và tải gói `SMPL_python_v.1.1.0.zip` theo hướng dẫn OmniRe. Đặt file `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` hoặc gói đã giải nén vào `MyDrive/SplatAD+OmniRe/`. Notebook tìm file Neutral trong `SMPL_python_v.1.1.0/smpl/models/`, kể cả khi gói có thêm một thư mục cùng tên bên ngoài. Notebook cũng nhận cùng bản v1.1 dưới tên `SMPL_NEUTRAL.pkl` ở thư mục dự án. [Trang tải SMPL](https://smpl.is.tue.mpg.de/download.php)
3. Mặc định `STAGE = "scene"` train SplatAD gốc, R2, R3 rồi OmniRe gốc trên cảnh `030` với sáu camera, 10.000 bước cho mỗi mô hình (`R3_STEPS` đổi riêng số bước của R3). R3 lưu checkpoint trong `runs/030-R3-gait/`, nên lượt R3 dùng MLP biến dạng trước đó không bị nạp nhầm. Checkpoint đã đủ bước được dùng lại. Mã run ghi `surround` cho sáu camera và `front3` cho ba camera trước, nên lượt ba camera cũ vẫn giữ nguyên trên Drive. `STAGE = "curriculum"` chạy cùng các mô hình qua `028 → 011 → 008`. Sửa `SCENES` để chọn cảnh. Các preset `smoke`, `pilot` và `confirm` lần lượt dùng 200, 1.200 và 4.000 bước. Tất cả các preset train RGB ở độ phân giải gốc từ bước 0. Các run Colab có `fullres` trong tên thư mục checkpoint. Lượt theo bộ đo và adapter v2 có `rgbq-v2` trong mã run.
4. Với mỗi cảnh, notebook xuất clip gốc và clip tái tạo `front-triptych.mp4` sau khi có checkpoint. Dashboard có các thẻ **Gốc**, **SplatAD**, **R2**, **R3** và **OmniRe**. Mỗi thẻ mô hình phát video ba camera trước render từ checkpoint cuối ở 60 FPS, trên cùng đồng hồ phát lại. Video OmniRe dùng cùng đường camera nội suy, còn trạng thái cảnh của OmniRe đổi theo frame 10 Hz vì OmniRe biểu diễn thời gian theo chỉ số frame. Mục **PSNR và SSIM** là bảng ba hàng người đi bộ, phương tiện, toàn cảnh và bốn cột SplatAD, R2, R3, OmniRe, mặc định ở bước cuối mà mọi mô hình cùng đo. Trong `MyDrive/SplatAD+OmniRe/`, `inputs/` và `pose-init/` chứa dữ liệu đã chuẩn bị, `runs/` chứa checkpoint, `artifacts/` chứa video, monitor và metric.

### OmniRe gốc

Cell **OmniRe gốc để đối chứng** dựng môi trường `.deps/omnire-env` (Python 3.10, PyTorch của Colab, requirements DriveStudio, gsplat v1.3.0, nvdiffrast, PyTorch3D, smplx). Sau đó `scripts/omnire-baseline.sh` chạy ba bước:

1. `prepare`: tiền xử lý PandaSet của DriveStudio (`images lidar calib pose dynamic_masks objects`), kiểm tra ID instance khớp với lượt pose, chép `smpl.pkl`, tạo mặt nạ bầu trời bằng SegFormer-B5 Cityscapes 1024 × 1024. Hướng dẫn gốc dùng mmseg với PyTorch 1.8, không chạy được trên Colab hiện tại, nên repo dùng cùng checkpoint qua Transformers.
2. `train`: vòng train lặp lại `tools/train.py` của DriveStudio theo đúng thứ tự. Repo lưu thêm `resume.pth` (mô hình, trạng thái Adam, bước, trạng thái ngẫu nhiên) mỗi 1.000 bước, nên lượt bị ngắt train tiếp từ đó; sau khi nạp, chỉ số k-NN của SMPL được dựng lại vì DriveStudio không lưu nó. Checkpoint gốc vẫn ghi mỗi 2.500 bước. Trong lúc train, gradient và loss được ghi mỗi 25 bước, PSNR/SSIM/L1 và mesh chiếu được đo mỗi 500 bước.
3. `video`: xuất video ba camera trước từ checkpoint cuối ở 60 FPS. Lệnh `score` chấm lại các checkpoint đã lưu khi cần.

### Khôi phục trên Colab

Sau khi Colab ngắt, mở lại notebook với cùng `STAGE`, danh sách cảnh và cấu hình run. Notebook kiểm tra môi trường, dùng lại pose và manifest đã lưu, rồi tiếp tục từ checkpoint hoàn chỉnh mới nhất của từng cảnh và biến thể. Những cảnh đã có checkpoint đủ bước và video được bỏ qua. Colab giải nén lại cảnh còn dở nếu runtime mới đã mất bộ nhớ đệm cục bộ.

- **Checkpoint.** Colab lưu theo chu kỳ 100 bước và ở bước cuối. Khi nhận tín hiệu dừng, trainer thử lưu ở cuối bước cập nhật hiện tại. Nếu runtime bị tắt ngay, lượt sau dùng checkpoint đã ghi xong gần nhất. File `.partial` và checkpoint ZIP bị cắt dở được bỏ qua.
- **Lỗi thư viện.** Notebook kiểm tra môi trường tách riêng cho train, 4D-Humans và chuyển SMPL. Nếu lệnh gặp lỗi import hoặc lỗi extension đã nhận diện, notebook dựng lại môi trường liên quan và thử một lần. Lượt train đọc lại checkpoint trước lần thử lại.
- **Tra lỗi.** `artifacts/<run_id>/progress/progress.json` ghi trạng thái và lần thử của từng giai đoạn. Các file `*.log` cùng thư mục ghi lệnh, output và lỗi. Khi lỗi vẫn còn, mở log của giai đoạn báo lỗi rồi chạy lại notebook sau khi sửa nguyên nhân.

### Theo dõi khi train

Cả bốn mô hình ghi vào `artifacts/<run_id>/030-<mô hình>-quality/`: `quality.json` (PSNR, SSIM và L1 của toàn cảnh, người đi bộ, phương tiện) và `training.jsonl` (loss từng thành phần và chuẩn gradient của pose SMPL, Gaussian SMPL, quỹ đạo hộp, Gaussian nền). Dashboard vẽ hai nhóm biểu đồ theo bước: L1/PSNR/SSIM của từng đối tượng, và gradient hoặc loss của từng mô hình. L1 vùng người là loss render của người đi bộ trên bộ ảnh đo cố định. Mục **Pose SMPL** chiếu mesh khởi tạo và mesh đang tinh chỉnh lên RGB gốc cho R2, R3 và OmniRe. Mỗi lượt chỉ giữ 3 checkpoint mới nhất trên Drive.

### PSNR và SSIM trong lúc train

- **SplatAD gốc.** Notebook chạy mô hình SplatAD tại commit đã pin trước mỗi cấu hình nghiên cứu. Hai bên dùng cùng cảnh, camera, số bước và ngân sách seed phù hợp với Colab. SplatAD dùng thiết lập actor và refinement của mô hình gốc. R3 giữ thiết lập này cho nền và xe.
- **Bộ đo cố định.** Bộ đo v2 lấy 8 ảnh cách đều cho mỗi camera train và bổ sung các quan sát người đi bộ đã chọn trong manifest. Mẫu bổ sung chỉ tính nhóm người đi bộ. PSNR và SSIM vùng người dùng bbox sát hơn với ngưỡng tối thiểu 11 × 11 px cho cửa sổ SSIM. Vùng ảnh dành cho LPIPS và overlay có thêm ngữ cảnh. Dashboard đặt kết quả người đi bộ lên đầu bảng, hiển thị số ảnh/crop và dùng cùng bộ đo cho mọi mốc. Mục Pose SMPL bổ sung số đo độ khớp mesh và cập nhật pose cho các track được theo dõi.
- **Hai mốc đối chiếu.** Lưu `initial.ckpt` trước lần cập nhật đầu tiên của run. Ghi `quality.json` ở bước 0, mỗi 250 lần cập nhật (50 với smoke) và bước cuối. Bảng hiển thị giá trị và chênh lệch so với SplatAD ở cùng bước, sau đó so với checkpoint đầu của run đang xem. Số liệu cập nhật khi Colab ghi mốc mới.

Bootstrap pin Python 3.12, PyTorch 2.7.1, CUDA toolkit 12.8/12.6 và các commit trong [versions.json](versions.json). Bootstrap ưu tiên `CUDA_HOME` nếu tương thích, rồi tìm các toolkit đã cài. Trên Colab Ubuntu 22.04 hoặc 24.04, bootstrap cài compiler và thư viện phát triển CUDA 12.8 nếu cần. Lần render đầu JIT-build fork GSplat. Giảm `--seed-points`, `--far-points` hoặc train ít camera hơn nếu GPU Colab thiếu VRAM. Dùng cùng các giá trị khi so sánh. [Mã nguồn SplatAD](https://github.com/georghess/neurad-studio/tree/aea389a936890eb48e1b43a493a52c0194559b48)

### Khởi tạo SMPL và lưu Drive

Notebook chạy OmniRe Option 2 để chiếu cuboid người lên ảnh, trích pose bằng 4D-Humans từ sáu camera, ghép tracklet và hoàn thiện các frame còn thiếu. Bản vá trích xuất ghi thêm mặt nạ pose trực tiếp cho monitor. Adapter đọc `smpl.pkl`, đổi hướng gốc sang hệ world và gắn UUID. Với frame được bù sau khi gộp camera, adapter nội suy lại hướng gốc trong hệ world và giữ góc khớp do OmniRe xuất. Tâm cuboid được nội suy giữa hai frame liền nhau trên lưới thời gian camera trước. Kết quả lưu trong `MyDrive/SplatAD+OmniRe/pose-init/omnire-option2/` theo phiên bản và mã băm của SMPL Neutral. [Chuẩn bị pose OmniRe](https://github.com/ziyc/drivestudio/blob/e59bda4fa681f829dbb1d65f0de582b0f633c450/docs/HumanPose.md)

- `OMNIRE_REFERENCE="off"` là mặc định và dùng kết quả trích xuất tại chỗ của OmniRe làm pose khởi tạo.
- `"compare"` ghi chênh lệch tịnh tiến, góc xoay gốc và góc xoay khớp trên các mốc chung với bản OmniRe công bố.
- `"fill"` dành cho lần chạy thử kết hợp hai nguồn. Chế độ này bù mốc còn trống khi bản công bố có pose cùng UUID và frame, rồi căn tâm mesh lại theo cuboid nếu shape khác nhau.
- Drive lưu `smpl.pkl` do OmniRe xuất, `tracks.json` dùng để train, manifest và báo cáo `comparison.json`. Khi bật đối chiếu hoặc kết hợp, Drive lưu thêm JSON của nguồn công bố và báo cáo ghi độ chênh pose trên các mốc chung.

### Quy ước và số liệu

- **Pose.** JSON gắn UUID sau khi đối chiếu ID nội bộ và sibling LiDAR. Rotation root của pose quan sát lấy từ camera 4D-Humans chọn. Hướng gốc của frame được bù dùng hệ world. Translation căn tâm bbox mesh đã skinning theo tâm cuboid tại timestamp của pose. Timestamp tính theo mốc giữa ảnh, xấp xỉ bằng timestamp camera PandaSet trừ 10 ms. Adapter đổi sang hệ cảnh và trừ epoch bằng độ chính xác đầy đủ trước khi cast float32.
- **Gaussian.** DA-pose canonical với trọng số LBS cố định, tâm tại vertex, quaternion theo pháp tuyến, scale theo diện tích và latent 13 kênh. Gaussian SMPL khởi tạo opacity riêng ở 0.99 theo OmniRe. R2 và R3 ghi checkpoint và metric dưới cùng mã run với SplatAD để đo từ bước 0. Tâm và hướng nhận gradient ở cả hai nhánh renderer. Vận tốc world nội suy từ biến đổi khớp lân cận. MCMC của SplatAD chạy trên bank nền và xe, còn số Gaussian SMPL cố định. Ở R2, UUID không có track SMPL đủ dùng đi theo actor rigid. Trong khoảng trống dài của track đã chọn, opacity của SMPL bằng 0.
- **Kiểm kê.** Cự ly là khoảng cách từ tâm camera đến tâm cuboid. Nhóm gần ≤15 m, trung bình 15–30 m, xa ≥30 m. Ngưỡng chiều cao bbox mặc định là 28 px. Bộ kiểm kê phân biệt pose trực tiếp từ 4D-Humans với pose do OmniRe nội suy. Ghi rõ trường che khuất và tay/chân nhìn thấy qua review thủ công.
- **RGB.** Render đủ nền, xe và người tại frame gốc. Tính MAE, PSNR, SSIM và LPIPS AlexNet cho toàn ảnh và vùng người từ bbox GT. MAE, PSNR và SSIM dùng crop tối thiểu 11 × 11 px. LPIPS và overlay dùng vùng ngữ cảnh tối thiểu 64 × 64 px. Trường hợp ảnh khớp hoàn toàn dùng PSNR 120 dB để lưu JSON.
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

- **Theo dõi mọi người đi bộ có SMPL.** Với R2 và R3, mỗi 250 bước monitor chọn một quan sát rõ nhất cho từng người (thêm các quan sát đã duyệt), chiếu mesh khởi tạo (cam) và mesh đang tinh chỉnh (xanh) lên RGB gốc, kèm crop render. Bảng **Vị trí pose so với khởi tạo** ghi độ dịch chuyển tịnh tiến gốc (cm), xoay gốc và xoay khớp (độ), trung bình và lớn nhất trên mọi frame có pose. OmniRe gốc được ghi cùng định dạng ở mỗi checkpoint 2.500 bước, trên cùng các quan sát. Dashboard có bộ chọn R2, R3 hoặc OmniRe.
- **PSNR và SSIM của bốn mô hình.** Mục PSNR và SSIM có thêm bảng SplatAD, R2, R3 và OmniRe cho toàn cảnh, người đi bộ và xe ở cùng bước, kèm chênh lệch so với SplatAD.

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
python -m research.keypoints --help  # cần transformers==4.53.2 trong train-env
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
