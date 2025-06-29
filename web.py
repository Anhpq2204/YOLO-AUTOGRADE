import streamlit as st
from PIL import Image
import yolo_model
import numpy as np
import cv2
import grade 
# # st.markdown(
# #     """
# #     <style>
# #     /* Cách cũ: với các class chung của Streamlit */
# #     .reportview-container, .main, .block-container {
# #         background-color: #FFFFFF;
# #     }
# #     /* Một vài class mới của Streamlit versions mới */
# #     .css-18e3th9 {
# #         background-color: #FFFFFF;
# #     }
# #     </style>
# #     """,
# #     unsafe_allow_html=True
# # )

# st.set_page_config(page_title="Demo Xử Lý Ảnh", layout="centered")

# img = Image.open("exam-sheet.png")
# st.image(img, use_column_width=True)

# st.title("Ứng dụng xử lý ảnh với Streamlit")


st.title('Ứng dụng chấm điểm THPTQG 2025')

# st.write('Tải ảnh của bạn lên tại đây:')

image=st.file_uploader("Tải ảnh của bạn lên tại đây")

if image is not None:
    st.write('Upload file thành công')
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Ảnh vừa được tải lên", use_column_width=True)
    image, final_string = yolo_model.solve(img_bgr)
    st.subheader("Thông tin phát hiện")
    st.write(final_string)
    diem=grade.grade_he(final_string)
    st.write(diem)










