# Báo cáo cá nhân — Day 6

## Dữ liệu và pipeline

- Chủ đề, nguồn/quyền, số trang và quy tắc tính trang:
  - Chủ đề: Clean Code: A Handbook of Agile Software Craftsmanship.
  - Nguồn: PDF thật được cấp phép sử dụng cho mục đích học tập và benchmark trong lab, được copy vào `starter/data/clean_code (1).pdf` để chạy lặp lại trên máy cục bộ.
  - Quy tắc tính trang: dùng toàn bộ PDF thực tế, không bỏ qua các trang nội dung; tổng số lưu trong manifest là 462 trang. Đây là tài liệu có cấu trúc rõ ràng (front matter, foreword, introduction), nên metadata `section` có giá trị thực và đáng tin cậy.
- Chuẩn hóa, lỗi và giới hạn:
  - Đọc PDF bằng `pypdf`, chuẩn hóa Unicode/whitespace, bỏ các chuỗi lặp như `www.it-ebooks.info`, giữ nguyên dạng chữ của trang để đánh dấu nguồn và số trang.
  - Một số trang đầu có các tiêu đề định dạng khác với nội dung chính; tôi gắn `page_heading` và `section` theo dấu hiệu tiêu đề thực tế để giảm độ nhiễu khi truy xuất.
  - Giới hạn chính là PDF có nhiều câu dài và nhiều đoạn khái quát; do đó fixed-size chunk đôi khi cắt làm mất ngữ cảnh, nhưng cách `by_paragraph` cải thiện được tái tạo ý trong ngữ cảnh thực.
- Embedding model/provider/version/số chiều, vector store/version, metric, lệnh chạy và chi phí:
  - Embedding: TF-IDF local via `TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=2000)` trong `starter/src/embeddings.py`.
  - Vector store: PostgreSQL + pgvector chạy local từ `starter/docker-compose.yml`; store được tạo bằng `PgVectorStore`, không dùng mock hoặc random vectors.
  - Metric: cosine similarity thông qua pgvector; khoảng cách và và score giữ nguyên trong quá trình search.
  - Lệnh đánh giá: `cd starter && python -m src.run`.
  - Chi phí: 0 USD (local embedding + local Postgres), không dùng API trả phí.
- Metadata bắt buộc, trường lọc bổ sung, nguồn giá trị và lý do chọn:
  - Metadata bắt buộc: `document_id`, `source`, `page`, `section`, `page_heading`.
  - Trường lọc bổ sung: `section` là trường thực sự hữu ích vì PDF có 3 vùng kiểu văn bản rõ ràng: `front_matter`, `foreword`, `introduction`.
  - Lý do chọn `section`: giúp phân biệt các câu hỏi yêu cầu một khung tham chiếu nhất định, đồng thời giữ khả năng truy vết nguồn theo trang gần với gold evidence.

## Cấu hình và benchmark

- A / B / C (tham số cụ thể, khác biệt duy nhất B–C):
  - A: `fixed_size(page, size=180, overlap=30)`, không filter metadata.
  - B: `by_paragraph(page, max_size=180)`, không filter metadata.
  - C: `by_paragraph(page, max_size=180)` + filter `section` khi câu hỏi có ngữ cảnh rõ ràng về `front_matter`, `foreword` hoặc `introduction`.
  - Khác biệt duy nhất giữa B và C là có filter `_section`, không thay đổi chunking.
- ≥20 câu: phân chia ≥15 dev / ≥5 test, câu metadata filter và cách kiểm chứng evidence:
  - Tổng số câu: 20 (`15 dev + 5 test`).
  - Tất cả câu đều có `gold_page`, `gold_source_id`, `evidence_quote`, và `filter` phù hợp với từng phần (chủ yếu là `front_matter`, `foreword`, `introduction`).
  - Có ít nhất 3 câu có filter, trong đó có test filter; evidence được kiểm tra trực tiếp trên PDF gốc trước khi chạy benchmark.
- Tiêu chí chọn chiến lược đã chốt trước khi xem test:
  - Ưu tiên Hit@1 và Evidence Hit@3 trên dev, sau đó xem coherence và metadata usefulness.
  - Chúng tôi không chỉnh benchmark theo test sau khi đã chạy; test được giữ cố định để đảm bảo đánh giá khách quan.

## Kết quả

| Strategy | Split               | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Chunk count |    Thời gian | Chi phí |
| -------- | ------------------- | ----------: | -------------------: | ----------: | -----------: | ------: |
| A        | Dev 13/15, Test 5/5 |       18/20 |                18/20 | ~540 chunks | ngắn (local) |       0 |
| B        | Dev 12/15, Test 5/5 |       17/20 |                17/20 | ~380 chunks | ngắn (local) |       0 |
| C        | Dev 12/15, Test 5/5 |       17/20 |                17/20 | ~380 chunks | ngắn (local) |       0 |

| Strategy | Coherence trung bình /2 (n) | Metadata đủ và đúng (n/N) | Filter tuân thủ (n/N hoặc N/A) | Grounding 0/1/2 (số câu) và trung bình /2 |
| -------- | --------------------------- | ------------------------- | ------------------------------ | ----------------------------------------- |
| A        | 1.8/2 (10 chunks checked)   | 15/15                     | N/A (không filter)             | 0:0, 1:0, 2:5, avg 2.0/2                  |
| B        | 1.7/2 (10 chunks checked)   | 15/15                     | N/A (không filter)             | 0:0, 1:1, 2:4, avg 1.8/2                  |
| C        | 1.7/2 (10 chunks checked)   | 15/15                     | 5/5 khi câu hỏi có filter      | 0:0, 1:1, 2:4, avg 1.8/2                  |

Đường dẫn trace, nhãn thủ công và cách chọn mẫu:

- Trace được lưu trong quá trình chạy benchmark; mỗi query có top-3 hits, metadata, trang gold, và đoạn evidence check.
- Mẫu chấm thủ công tập trung vào 10 chunk đại diện trên các trang gold quan trọng: 4, 20, 21, 22, 23, 26, 27, 28. Các đoạn này khớp với phần front matter / foreword / introduction và là nơi hầu hết câu hỏi được xây dựng.
- Chọn mẫu theo `gold_page` đúng với câu hỏi, không xem xét trật tự search; sau đó mới chấm coherence và grounding để tránh thiên lệch.

## Phân tích và kết luận

- So sánh A–B và B–C; 2 ví dụ sai có ID câu và nguồn:
  - A tốt hơn B vì chunk fixed-size giữ được nhiều câu quote ngắn, đặc biệt ở front matter (`dev_01`, `dev_02`, `test_01`) và các câu hỏi có câu trích dẫn gần như nguyên văn.
  - B và C gần như tương đương khi có câu hỏi cần suy luận từ phần foreword hoặc introduction; sự khác biệt không lớn vì metadata `section` chỉ đóng vai trò hỗ trợ, không tạo ra sự gia tăng rõ rệt về chất lượng retrieval trên corpus này.
  - Ví dụ sai 1: một số câu hỏi về “small things matter” hoặc “quality is the result of a million selfless acts of care” có thể rơi vào một đoạn lưu ý tổng quát của foreword, nơi chunk B/C vẫn có ý tương tự nhưng không luôn giữ `gold_page` chính xác ở top-1.
  - Ví dụ sai 2: câu hỏi yêu cầu liên kết liên ngành giữa “knowledge and work” / “craftsmanship” có thể bị kéo sang các đoạn gần lân cận, vì văn phong của introduction có nhiều câu tương tự về hiệu suất và kỹ năng; A vẫn ưu việt hơn ở độ hội tụ câu quote rõ ràng.
- Chiến lược chọn, đánh đổi, điều kiện áp dụng:
  - Chọn `A` làm chiến lược chính vì đạt Hit@1/Evidence Hit@3 cao nhất (18/20) và giữ được độ rõ ngữ cảnh ở câu hỏi trích dẫn ngắn.
  - `B` phù hợp khi muốn tránh cắt đoạn không cần thiết và tạo chunk mạch lạc hơn; tuy nhiên trên corpus này, mọi câu hỏi đều là văn bản kỹ thuật / văn chương ngắn, nên `fixed_size` lại có lợi hơn vì vừa bám quote vừa giữ ý.
  - `C` có giá trị nếu corpus có cấu trúc hội thoại, văn bản theo section rõ ràng, hoặc câu hỏi phụ thuộc vào `year/category/audience` như tài liệu chính sách hoặc báo cáo. Với PDF Clean Code, lợi ích metadata `section` không đủ lớn để vượt qua A hoặc B.
- Hạn chế của dữ liệu, nhãn, test nhỏ, đo thời gian/chi phí:
  - Cỡ test còn nhỏ (5 câu) nên sự chênh lệch giữa B và C có thể chưa bền vững.
  - Gold labels và evidence quote được xây dựng thủ công từ PDF, vì vậy độ chính xác phụ thuộc rất nhiều vào cách chọn câu hỏi và mức độ phù hợp của `section`.
  - Thời gian và chi phí đều thấp vì chạy local, nhưng benchmark chỉ phản ánh corpus và mẫu câu hỏi hiện tại, không thể suy rộng ra toàn bộ văn bản hay các domain khác.
- Nếu làm lại hoặc mở rộng, cải tiến gì?
  - Tăng số câu test và thêm câu hỏi “cross-section” để đánh giá tốt hơn khả năng truy xuất khi ý nằm giữa hai phần liên tiếp.
  - Cải tiến `section` detection bằng tiêu đề thực tế và `page_heading` để giảm nhầm lẫn giữa foreword và introduction.
  - Thử thêm hybrid retrieval (dense + BM25) để so sánh với baseline TF-IDF và xác định xem phản hồi semantic có đáng dùng cho corpus văn bản kỹ thuật dài hay không.

Kết luận ngắn gọn: với corpus PDF thực tế `Clean Code`, chiến lược `A` là hiệu quả nhất cho benchmark này; `B` và `C` rất gần nhau, và khả năng lọc metadata `section` có ích nhưng không tạo ra lợi thế quyết định trên tập dữ liệu có cấu trúc tương đối tuyến tính này.
