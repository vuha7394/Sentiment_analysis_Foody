# Business Objective/Problem
● Foody.vn là một kênh phối hợp với các nhà hàng/quán ăn bán thực phẩm online.
● Chúng ta có thể lên đây để xem các đánh giá, nhận xét cũng như đặt mua thực phẩm.
● Từ những đánh giá của khách hàng, vấn đề được đưa ra là làm sao để các nhà hàng/ quán ăn hiểu được khách hàng rõ hơn, biết họ đánh giá về mình như thế nào để cải thiện hơn trong dịch vụ/sản phẩm.

## Các kiến thức/ kỹ năng cần để giải quyết vấn đề này:
● Hiểu vấn đề
● Import các thư viện cần thiết và hiểu cách sử dụng
● Đọc dữ liệu (dữ liệu project này được cung cấp)
● Thực hiện EDA cơ bản (sử dụng Pandas Profiling Report)
● Tiền xử lý dữ liệu: làm sạch, tạo tính năng mới, lựa chọn tính năng cần thiết...
● Trực quan hóa dữ liệu
● Lựa chọn thuật toán cho bài toán classification
● Xây dựng model
● Đánh giá model
● Báo cáo kết quả

Dữ liệu là những bình luận được thu thập từ Foody. Mỗi bình luận được gắn nhãn từ 1 đến 5 sao. Vì mục tiêu của dự án này là phân loại cảm xúc nên chúng tôi sẽ gắn nhãn lại các nhận xét thành 3 lớp theo quy tắc sau:
- 1-2-3 sao: 'tiêu cực'
- 4-5 sao: 'tích cực'

## Bước 1: Business Understanding
Dựa vào mô tả nói trên => xác định vấn đề:
- Xây dựng hệ thống dựa trên lịch sử những đánh giá của khách hàng đã có trước đó. Dữ liệu được thu thập từ phần bình luận và đánh giá của khách hàng ở Foody.vn...
- Mục tiêu/ vấn đề: Xây dựng mô hình dự đoán giúp nhà hàng có thể biết được những phản hồi nhanh chóng của khách hàng về sản phẩm hay dịch vụ của họ (tích cực, tiêu cực hay trung tính), điều này giúp cho nhà hàng hiểu được tình hình kinh doanh, hiểu được ý kiến của khách hàng từ đó giúp nhà hàng cải thiện hơn trong dịch vụ, sản phẩm.

## Bước 2: Data Understanding/ Acquire
Từ mục tiêu/ vấn đề đã xác định: xem xét các dữ liệu cần thiết:
- Dữ liệu không có sẵn > Cần lên Foody.vn để thu thập dữ liệu lịch sử đánh giá của khách hàng
- Sau khi thu dữ liệu, ta có: 45.000 mẫu gồm 3 thông tin: Tên nhà hàng, nội dung review và điểm đánh giá.

Có thể tập trung giải quyết bài toán
- Sentiment analysis in Cuisine Area với các thuật toán thuộc nhóm Supervised Learning – Classification như: Naïve Bayes, KNN, Logistic Regression...
- Gắn nhãn và chia nhóm theo nhóm đánh giá sau:
  - 1-2-3 sao: 'negative'
  -  4-5 sao: 'positive'
