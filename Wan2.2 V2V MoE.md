2. Luồng Dữ Liệu Inference Mới Của Hệ Thống┌── PHASE 1 & 2: DATA LOADING & PREPROCESSING ──────────────────────────┐
│  - Tải 16 frames history gốc từ test.jsonl                           │
│  - Chuẩn hóa text qua build_wan_prompt()                                     │
│  - Đổi kích cỡ ảnh về dạng native [1, 16, 3, 480, 832]                      │
└───────────────────────────────────┬───────────────────────────────────┘
                                    ▼
┌── PHASE 3 & 4: NATIVE V2V MOE INFERENCE ──────────────────────────────┐
│  - Nạp text embedding [B, 77, 768] và video_input [1, 16, 3, 480, 832]│
│  - Khởi chạy vòng lặp Flow-Matching 50 bước qua bộ giải Euler solver  │
│  - Timestep t > 500: Gọi High-Noise Expert định hình phông nền/vật thể │
│  - Timestep t <= 500: Gọi Low-Noise Expert tinh chỉnh chi tiết ảnh    │
└───────────────────────────────────┬───────────────────────────────────┘
                                    ▼
┌── PHASE 5: ADVANCED INFERENCE GUIDANCE (ECCV NOVELTY) ────────────────┐
│  - Thực thi tính toán Latent Flow Guidance tại mỗi bước t            │
│  - Phân bổ chuyên gia MoE động dựa trên nhãn Semantic Phase Label     │
└───────────────────────────────────┬───────────────────────────────────┘
                                    ▼
┌── PHASE 6, 7 & 8: POST-PROCESSING & SAVE ─────────────────────────────┐
│  - Giải mã qua Wan2.2-VAE Decoder                                             │
│  - Chạy cụm RAFT Optical Flow + RANSAC Homography vá nền tĩnh       │
│  - Phóng đại độ phân giải lên chuẩn thi đấu [1280 × 720] PNG        │
└───────────────────────────────────────────────────────────────────────┘
3. Các Tính Năng Đột Phá Cần Lập Trình (ECCV Features)Tính năng 1: Inference-time Latent Flow Guidance (Dẫn hướng Luồng Quang Học trong Không Gian Ẩn)Yêu cầu cho Agent: Thay vì chỉ vá nền ở tầng pixel sau khi decode, Agent phải cài đặt một cơ chế tính toán đạo hàm sai lệch chuyển động (Loss Gradient) ngay trong vòng lặp khử nhiễu.Công thức Toán học áp dụng: Tại mỗi bước thời gian $t$, từ trạng thái latent hiện tại $x_t$, mô hình dự báo cấu trúc latent sạch tạm thời $\hat{x}_0$. Gọi $\mathcal{F}$ là hàm trích xuất trường luồng quang học (Flow Field Vector) và $\mathcal{H}(V_{\text{hist}})$ là ma trận biến đổi phối cảnh ổn định tính từ 16 khung hình lịch sử gốc. Hàm mục tiêu ràng buộc hình học (Geometric Loss) được định nghĩa:$$\mathcal{L}_{\text{flow}} = \|\mathcal{F}(\hat{x}_0) - \mathcal{H}(V_{\text{hist}})\|_2^2$$Cơ chế cập nhật: Vector nhiễu dự báo $\epsilon$ sẽ được điều chỉnh bằng Gradient của $\mathcal{L}_{\text{flow}}$ nhân với trọng số điều hướng $\lambda$:$$\hat{\epsilon} = \epsilon - \lambda \nabla_{x_t} \mathcal{L}_{\text{flow}}$$Tính năng 2: Phase-Aware Semantic Expert Routing (Điều hướng Chuyên gia theo Pha Hành vi)Yêu cầu cho Agent: Can thiệp vào cổng chọn chuyên gia (Gating/Routing Network) mặc định của Wan2.2 MoE.Đọc nhãn phase_label từ file manifest (0: pre-recognition, 1: recognition, 2: judgment, 3: action, 4: avoidance).Nếu phase_label $\in \{3, 4\}$ (pha hành động gắt/né tránh hiểm họa): Ép mạng lưới tăng cường trọng số tương tác cho các chuyên gia phụ trách biến động động học lớn (Temporal Experts).Nếu phase_label $\in \{0, 1\}$ (pha đi đều/nhận diện tĩnh): Tăng trọng số cho các chuyên gia bảo tồn chi tiết cấu trúc tĩnh không gian (Spatial Experts).4. Đặc Tả Từng File Code Cần Sửa Đổi Cho AI AgentNhiệm vụ 1: Sửa đổi file src/pipelines/wan2_2_v2v_pipeline.pyXóa bỏ cơ chế gộp chuỗi thủ công của Wan2.1. Triển khai class gọi mô hình MoE V2V chính thống của Wan2.2 và tích hợp Latent Flow Guidance.Python# THIẾT KẾ MẪU KHUNG CODE DÀNH CHO AGENT
import torch
import torch.nn.functional as F
from diffusers import Wan2_2MoEBlendedPipeline # Thay thế lớp pipeline 2.1 cũ

class Wan22V2VSOTAInferencePipeline:
    def __init__(self, model_id, device):
        self.pipe = Wan2_2MoEBlendedPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        self.pipe = self.pipe.to(device)
        self.pipe.eval()
        self.device = device
        
    def latent_flow_guidance_step(self, latent_pred, history_flow_map, lambda_weight=0.1):
        """
        [LẬP TRÌNH TÍNH NĂNG ECCV 1]
        Tính toán Loss giữa luồng quang học của Latent dự báo với ma trận dịch chuyển gốc
        Trả về vector Gradient để hiệu chỉnh hướng khử nhiễu
        """
        latent_pred.requires_grad_(True)
        # 1. Giả định trích xuất flow thô từ latent_pred
        # 2. Tính toán L2 Loss với history_flow_map
        # 3. Tính toán gradient = torch.autograd.grad(loss, latent_pred)
        # 4. Trả về latent_pred - lambda_weight * gradient
        pass

    @torch.no_grad()
    def generate_video(self, prompt, video_input, phase_label, config):
        """
        Hàm sinh video chính thức sử dụng Native V2V và Semantic Routing
        """
        # [LẬP TRÌNH TÍNH NĂNG ECCV 2]
        # Thực hiện can thiệp router_logits dựa trên phase_label trước khi đưa vào transformer blocks
        
        # Thực hiện vòng lặp khử nhiễu tích hợp Latent Flow Guidance qua từng bước Euler
        # ...
        pass
Nhiệm vụ 2: Tạo mới hoặc chỉnh sửa file scripts/infer/infer_wan2_2.pyTạo script thực thi chính, kế thừa toàn bộ cấu trúc nạp dữ liệu cũ của file infer_wan.py. Tuy nhiên, thay đổi hoàn toàn tham số truyền vào mô hình thành cấu trúc Native V2V và giữ lại module hậu xử lý bảo tồn nền bằng thuật toán truyền thống.Python# HƯỚNG DẪN FLOW CHẠY CHO AGENT
# 1. Khởi tạo parser nhận diện các tham số mới: --model_id (Mặc định: Wan-AI/Wan2.2-V2V-14B-MoE), --flow_shift (Mặc định: 3.0)
# 2. Đọc file JSONL manifest chuẩn của cuộc thi Track 5
# 3. Chuẩn hóa prompt thông qua module src.utils.prompt_utils.build_wan_prompt
# 4. Gọi Wan22V2VSOTAInferencePipeline để sinh chuỗi ảnh thô
# 5. Đẩy chuỗi ảnh thô qua module src.utils.flow_utils.preserve_background để xử lý Alpha Compositing
# 6. Ép phóng đại độ phân giải ảnh về 1280x720 và lưu định dạng lossless PNG
5. Các Ràng Buộc Kiểm Tra Nghiêm Ngặt (Guardrails cho Agent)RÀNG BUỘC 1 (ĐÓNG BĂNG TRỌNG SỐ): Đảm bảo toàn bộ cấu phần của Wan2.2 MoE (Text Encoder, VAE, DiT) đều phải chạy ở chế độ .eval() và được bọc trong block with torch.no_grad(): ngoại trừ bước tính toán Latent Flow Guidance. TUYỆT ĐỐI KHÔNG thực hiện cập nhật trọng số mô hình gốc (Không fine-tune).RÀNG BUỘC 2 (ĐỘ PHÂN GIẢI ĐẦU RA): Toàn bộ ảnh lưu vào thư mục prediction_wan2.2_sota bắt buộc phải được nội suy resize về đúng kích thước chuẩn bài thi $1280 \times 720$.RÀNG BUỘC 3 (QUẢN LÝ VRAM): Vì đây là cấu trúc MoE, cần kiểm tra và giải phóng bộ nhớ đệm liên tục bằng lệnh torch.cuda.empty_cache() sau khi xử lý xong mỗi mẫu dữ liệu để tránh tràn bộ nhớ card A6000.Yêu cầu dành cho AI Agent: Hãy đọc kỹ toàn bộ sơ đồ cấu trúc file và thực hiện việc lập trình mã nguồn một cách tuần tự, chính xác.

link github: https://github.com/Wan-Video/Wan2.2.git

Ngoài ra, còn có bộ dataset bdd5k, cũng có thể tận dụng
Cách 1: Dùng BDD5K để huấn luyện "Semantic Gate" (Novelty số 2)
Nếu bạn chọn hướng làm Phase-Aware Semantic Expert Routing (Điều hướng chuyên gia MoE theo hành vi lái xe), bạn hãy dùng BDD5K làm tập huấn luyện bổ trợ cho mạng Gating mạng phụ (nhánh tuyến tính siêu nhẹ).

Cách làm: Bạn dùng một mô hình ngôn ngữ hoặc luật để tự động gán nhãn hành vi cho BDD5K. Sau đó, dùng cả WTS và BDD5K để train bộ Semantic Gate.

Đây là các novelty:
1. Inference-time Latent Flow Guidance (Đẩy Luồng Quang Học vào Không Gian Ẩn)Sự hạn chế hiện tại:Pipeline hiện tại đang tính toán Luồng quang học (Optical Flow) và trộn Alpha Compositing ở tầng Hậu xử lý pixel (sau khi VAE đã decode xong). Cách làm này mang tính chắp vá: nếu mô hình sinh ra vật thể bị lỗi hoặc sai cấu trúc hình học, việc vá phông nền ở bước cuối có thể tạo ra các bóng mờ (ghosting artifacts) hoặc đường ranh giới không tự nhiên xung quanh xe cộ.Ý tưởng Novelty cho Paper:Thay vì xử lý sau cùng, bạn hãy biến trường luồng quang học (Velocity/Flow Field) thành một hàm mục tiêu dẫn đường (Guidance Gradient) ngay trong quá trình khử nhiễu từng bước của Flow-Matching.[Nhiễu x_t] ──► [Mạng DiT] ──► [Dự báo Latent]
                     ▲
                     │ (Cộng thêm Gradient sửa sai)
         [Bộ tính toán Latent Flow Loss] ◄── [Ma trận Homography từ 16 frames gốc]
Cơ chế toán học: Tại mỗi bước giải Euler solver, bạn chiếu Latent dự báo tạm thời qua một tầng tính toán Flow Field thu nhỏ. Sau đó tính khoảng cách toán học (Loss) giữa trường chuyển động này với ma trận biến đổi phối cảnh (Homography) trích xuất từ 16 khung hình lịch sử gốc.Hành động: Tính đạo hàm của hàm Loss này đối với Latent hiện tại và dùng Gradient đó để tinh chỉnh trực tiếp vector khử nhiễu ($x_{t-1}$).Giá trị học thuật: Bạn có thể đặt tên phương pháp là "Physics-conditioned Latent Flow Matching". Bài báo sẽ chứng minh bằng toán học rằng việc ép ràng buộc vật thể chuyển động tuân thủ định luật hình học ngay trong không gian ẩn giúp mô hình sinh ra video có độ mượt và tính nhất quán cấu trúc vượt trội mà không cần huấn luyện lại.2. Phase-Aware Semantic Expert Routing (Định hướng Chuyên gia theo Hành vi Lái xe)Sự hạn chế hiện tại:Kiến trúc MoE (Mixture-of-Experts) của Wan2.2 mặc định chỉ phân chia chuyên gia một cách máy móc theo thời gian (bước nhiễu cao/thấp) hoặc các đặc trưng toán học thuần túy của ảnh. Nó chưa hề hiểu bản chất ngữ nghĩa của các pha hành vi giao thông phức tạp.Ý tưởng Novelty cho Paper:Tập dữ liệu WTS cung cấp các nhãn phân kỳ hành vi (phase_label) cực kỳ tường minh: từ chuẩn bị nhận diện (pre-recognition), đưa ra quyết định (judgment) cho đến thực hiện hành động tránh né (avoidance/outcome). Bản chất chuyển động của xe cộ ở các pha này hoàn toàn khác nhau (Pha Judgment xe thường đi đều, pha Action/Avoidance xe sẽ bẻ lái gắt hoặc phanh đột ngột).Cơ chế kiến trúc: Thiết kế một mạng nơ-ron phụ tuyến tính siêu nhẹ (gọi là Semantic Gate). Mạng này nhận đầu vào là mã hóa của phase_label và sample_type (góc camera) để tính toán ra một vector trọng số phân bổ chuyên gia.Hành động: Bơm trực tiếp vector trọng số này vào hàm chọn chuyên gia (Routing Function) của Wan2.2. Nghĩa là: khi gặp pha nguy hiểm (avoidance), hệ thống sẽ ép mạng lưới tự động kích hoạt các tầng Attention chuyên xử lý biến động thời gian lớn; khi gặp pha đi đều (pre-recognition), mô hình sẽ kích hoạt các chuyên gia mạnh về giữ vững chi tiết không gian.Giá trị học thuật: Tên phương pháp gợi ý: "Semantic-Driven Mixture-of-Experts for Event-driven Video Forecasting". Bạn chứng minh được rằng việc dẫn hướng chuyên gia bằng ngữ nghĩa hành vi thế giới thực (Real-world Driving Semantics) giúp mô hình dự báo chính xác các bước ngoặt bất ngờ của giao thông.3. Object-Centric Slot Attention Conditioning (Căn chỉnh Điều kiện theo Từng Vật thể)Sự hạn chế hiện tại:UMT5 mã hóa toàn bộ câu Caption thành một khối vector phẳng $[B, 77, 768]$ rồi đưa vào mạng qua Cross-Attention. Điều này khiến mô hình dễ bị hiện tượng "lẫn lộn thuộc tính" (Attribute Binding Problem). Ví dụ câu lệnh ghi "Dark blue sedan driving fast, pedestrian waiting", mô hình có thể vô tình vẽ người đi bộ màu xanh hoặc chiếc xe bị biến dạng theo chuyển động của người.Ý tưởng Novelty cho Paper:Thay vì bắt mạng DiT đọc một cục văn bản thô, bạn tách câu Caption thành các thực thể độc lập (Object-centric Slots) dựa trên checklist thuộc tính của bộ dữ liệu WTS.[WTS Caption] ──► [LLM/Parser] ──► Slot 1: Car (Dark blue, Fast)
                               ──► Slot 2: Pedestrian (Waiting)
                                             │
                                             ▼
                                  [Slot Attention Blocks] ──► Inject into DiT
Cơ chế kiến trúc: Sử dụng một cấu trúc mạng Adapter nhỏ (dựa trên thuật toán Slot Attention kinh điển). Mỗi "Slot" sẽ chịu trách nhiệm học một thực thể (Xe A, Xe B, Người đi bộ C, Phông nền đường) kèm theo các thuộc tính động học của nó.Hành động: Thay thế tầng Cross-Attention mặc định bằng Object-to-Patch Cross-Attention. Mô hình sẽ tính toán mức độ tương tác giữa từng Patch không gian trên ảnh với từng Slot vật thể cụ thể.Giá trị học thuật: Tên phương pháp gợi ý: "Compositional Video Forecasting via Attribute-bound Slot Attention". Phương pháp này giải quyết triệt để bài toán đồng bộ hóa ngữ nghĩa và không gian ảnh, giúp điểm số CLIP-S (độ khớp văn bản - video) và FVD (độ tự nhiên của video) tăng vọt vượt trội so với các mô hình sinh video thông thường.