import cv2
import numpy as np

# 先读取学校的校徽
gdut_logo = cv2.imread('D:\\image\\GDUT.png')

# 加载人脸识别分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 再进行视频捕捉
cap = cv2.VideoCapture(0)

while True:
    # 读取视频帧
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 人脸检测
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # 将校徽调整为适应人脸
        gdut_logo_resized = cv2.resize(gdut_logo, (w, h))

        # 人脸四个顶点
        pts_dst = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype='float32')

        # 校徽四个顶点
        pts_src = np.array([[0, 0], [gdut_logo_resized.shape[1], 0],
                            [gdut_logo_resized.shape[1], gdut_logo_resized.shape[0]],
                            [0, gdut_logo_resized.shape[0]]], dtype='float32')

        # 透视变换矩阵
        H = cv2.getPerspectiveTransform(pts_src, pts_dst)

        # 对GDUT.png进行透视变换
        warped_logo = cv2.warpPerspective(gdut_logo_resized, H, (frame.shape[1], frame.shape[0]))

        # 创建遮罩
        # mask = np.zeros_like(frame, dtype=np.uint8)
        # mask = cv2.fillConvexPoly(mask, pts_dst.astype(int), (255, 255, 255))

        # 将校徽叠加到原图上
        frame = cv2.addWeighted(frame, 1, warped_logo, 0.7, 0)

    # 显示处理后的图像
    cv2.imshow('AR GDUT Logo', frame)

    # 按下 'e' 退出
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
