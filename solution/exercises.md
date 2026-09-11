# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)

    Khi temperature tăng từ 0.0 lên 1.5, mức độ sáng tạo, ngẫu nhiên và đa dạng của từ vựng trong câu trả lời tăng rõ rệt. Ở mức 0.0, mô hình luôn trả về kết quả cố định, mang tính khuôn mẫu cao trong khi ở mức 1.5, câu trả lời trở nên phong phú hơn nhưng dễ xuất hiện lỗi lạc đề hoặc thông tin kém chính xác.
   
### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**

    Chatbot hỗ trợ khách hàng cần sự chính xác, nhất quán và bám sát tài liệu hướng dẫn hoặc quy định của công ty, hạn chế tối đa việc mô hình tự bịa đặt (hallucination) thông tin. Nên emm chon từ 0.2-0.3

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**

    Tổng số request mỗi ngày = 10.000 { người dùng} x 3  lần/ngày = 30.000 request/ngày$.Giả định token đầu vào tương đương token đầu ra hoặc ước lượng dựa trên tỷ lệ chuẩn (ví dụ: mỗi request tiêu thụ trung bình 100 token input và 350 token output).
    Chi phí cho GPT-4o đắt hơn GPT-4o-mini khoảng 16,6 lần cho cùng một khối lượng workload đầu ra (với giá output tương ứng là $0.010 so với $0.0006 mỗi 1K token)
    Trường hợp GPT-4o xứng đáng: Các tác vụ lập luận phức tạp đa bước, phân tích mã nguồn nâng cao, hoặc dịch thuật văn học đòi hỏi sắc thái ngôn ngữ tinh tế.  
    Trường hợp nên dùng mini: Các tác vụ phân loại ý định (intent classification), trích xuất thông tin đơn giản, tóm tắt văn bản hàng loạt hoặc chatbot hỏi đáp thông thường có lưu lượng truy cập lớn


## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)

    Sự khác biệt: Phản hồi từ giáo viên tiểu học có độ dài ngắn, dùng từ ngữ thân thuộc, gần gũi và ví dụ minh họa bằng trò chơi hoặc đồ vật thực tế (như chuỗi hạt, khối Lego). Ngược lại, phản hồi từ chuyên gia tài chính dài hơn, dày đặc thuật ngữ kỹ thuật chuyên ngành (như cơ chế đồng thuận, mã hóa băm, sổ cái phi tập trung) và cấu trúc logic phức tạp.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**

    Khảo sát số liệu: Lấy một đoạn văn mẫu tiếng Việt gồm 100 từ:Theo công thức ước lượng (Số từ / 0.75$): 100 / 0.75 = 133 token.Theo count_tokens thực tế bằng tiktoken (bộ mã hóa o200k_base hoặc cl100k_base), một đoạn văn 100 từ tiếng Việt thường tiêu tốn khoảng từ 180 đến 220 token (chênh lệch khoảng 35% đến 65% cao hơn so với mức ước lượng cơ bản).Nguyên nhân tiếng Việt tốn nhiều token: Các bộ tách token (tokenizer) được huấn luyện chủ yếu dựa trên từ vựng và kho ngữ liệu tiếng Anh. Tiếng Việt chứa nhiều dấu thanh, ký tự Latin mở rộng và các từ ghép mà bộ tách từ không nhận diện trọn vẹn thành một token nguyên vẹn, buộc mô hình phải bẻ nhỏ một từ đơn thành nhiều mảnh token phụ (subword tokens) khác nhau


## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)

    Streaming quan trọng nhất trong các ứng dụng giao tiếp thời gian thực như chatbot hội thoại hoặc trợ lý ảo, giúp giảm đáng kể thời gian phản hồi cảm nhận (perceived latency) bằng cách hiển thị từng phần văn bản ngay khi mô hình sinh ra. Ngược lại, non-streaming phù hợp hơn cho các tác vụ tự động hóa ngầm, gọi API lập trình (API-to-API), hoặc khi ứng dụng yêu cầu toàn bộ đầu ra phải hoàn chỉnh tuyệt đối trước khi xử lý tiếp (như phân tích cấu trúc JSON, dịch thuật tài liệu định dạng phức tạp).

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**


    Lợi thế của Exponential Backoff: Giảm tải hiệu quả cho hệ thống đang quá tải bằng cách nới rộng khoảng thời gian chờ sau mỗi lần thất bại liên tiếp (ví dụ: 1s, 2s, 4s, 8s...), cho phép server có không gian và thời gian để hồi phục.

    Hậu quả của delay cố định: Nếu hàng nghìn client cùng retry với một khoảng thời gian cố định giống nhau, hiện tượng "thundering herd" (quá tải đồng loạt) sẽ xảy ra khi toàn bộ client đồng loạt gửi lại request cùng một thời điểm, khiến server tiếp tục sập vòng lặp vô tận.

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**

    Persona: Trợ lý Kỹ thuật Robot & Hệ thống Nhúng chuyên nghiệp.

    System Prompt: "Bạn là kỹ sư trưởng giàu kinh nghiệm về Robotics và hệ thống nhúng. Hãy tư vấn giải pháp bằng cấu trúc chuẩn, súc tích, đi thẳng vào bản chất kỹ thuật, ưu tiên mã nguồn C++/Python và ROS 2 chuẩn mực. Giao tiếp hoàn toàn bằng tiếng Việt chuyên ngành."

    Giải thích từ ngữ quan trọng:

    "Đi thẳng vào bản chất kỹ thuật": Giúp loại bỏ hoàn toàn các lời mở đầu xã giao rườm rà, tập trung cao độ vào công thức, sơ đồ khối hoặc mã nguồn mà một kỹ sư hoặc lập trình viên cần.

    "Chỉ định ngôn ngữ tiếng Việt chuyên ngành": Đảm bảo các thuật ngữ cốt lõi (như SLAM, Navigation2, node, topic) được giữ nguyên chuẩn quốc tế nhưng diễn giải bằng văn phong chuyên nghiệp, phù hợp với môi trường làm việc kỹ thuật tại Việt Nam.
 
### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**

    Hạn chế lớn nhất: Trợ lý hiện thiếu bộ nhớ dài hạn (long-term memory) và không có ngữ cảnh xuyên suốt giữa các phiên làm việc, dẫn đến việc phải cấu hình lại hoặc giải thích lại các thông số phần cứng (như kích thước robot, giới hạn vận tốc, cấu hình sensor) ở mỗi lần mở phiên chat mới.

    Đề xuất cải thiện: Triển khai cơ chế Retrieval-Augmented Generation (RAG) kết hợp bộ nhớ vector (như ChromaDB hoặc FAISS) lưu trữ tài liệu dự án cá nhân, sơ đồ CAD và mã nguồn trước đó của người dùng.

    Cách triển khai cụ thể: Khi người dùng đặt câu hỏi, hệ thống sẽ nhúng (embed) câu hỏi đó, tìm kiếm trong cơ sở dữ liệu vector các tài liệu kỹ thuật hoặc đoạn mã liên quan nhất từ các phiên làm việc trước, sau đó tiêm (inject) các đoạn ngữ cảnh đó vào system prompt dưới dạng context để mô hình đưa ra câu trả lời chính xác, bám sát đúng tiến độ phần cứng hiện tại.


## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
