import cv2
import gradio as gr
import mediapipe as mp
import dlib
import imutils
import numpy as np
from torchlm.tools import faceboxesv2
from torchlm.models import pipnet
import torchlm



mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

def apply_media_pipe_face_detection(image):
    with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.detections:
            return image
        annotated_image = image.copy()
        for detection in results.detections:
            mp_drawing.draw_detection(annotated_image, detection)
        return annotated_image


def apply_media_pipe_facemesh(image):

    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return image
        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
              image=annotated_image,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_TESSELATION,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
              image=annotated_image,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_CONTOURS,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
              image=annotated_image,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_IRISES,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_iris_connections_style())
            return annotated_image


class FaceOrientation(object):
    def __init__(self):
        self.detect = dlib.get_frontal_face_detector()
        self.predict = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

    def create_orientation(self, frame):
        draw_rect1 = True
        draw_rect2 = True
        draw_lines = True

        frame = imutils.resize(frame, width=800)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = self.detect(gray, 0)

        for subject in subjects:
            landmarks = self.predict(gray, subject)
            size = frame.shape

            # 2D image points. If you change the image, you need to change vector
            image_points = np.array([
                (landmarks.part(33).x, landmarks.part(33).y),  # Nose tip
                (landmarks.part(8).x, landmarks.part(8).y),  # Chin
                (landmarks.part(36).x, landmarks.part(36).y),  # Left eye left corner
                (landmarks.part(45).x, landmarks.part(45).y),  # Right eye right corne
                (landmarks.part(48).x, landmarks.part(48).y),  # Left Mouth corner
                (landmarks.part(54).x, landmarks.part(54).y)  # Right mouth corner
            ], dtype="double")

            # 3D model points.
            model_points = np.array([
                (0.0, 0.0, 0.0),  # Nose tip
                (0.0, -330.0, -65.0),  # Chin
                (-225.0, 170.0, -135.0),  # Left eye left corner
                (225.0, 170.0, -135.0),  # Right eye right corne
                (-150.0, -150.0, -125.0),  # Left Mouth corner
                (150.0, -150.0, -125.0)  # Right mouth corner

            ])
            # Camera internals
            focal_length = size[1]
            center = (size[1] / 2, size[0] / 2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype="double"
            )

            dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                          dist_coeffs)

            (b1, jacobian) = cv2.projectPoints(np.array([(350.0, 270.0, 0.0)]), rotation_vector, translation_vector,
                                               camera_matrix, dist_coeffs)
            (b2, jacobian) = cv2.projectPoints(np.array([(-350.0, -270.0, 0.0)]), rotation_vector,
                                               translation_vector, camera_matrix, dist_coeffs)
            (b3, jacobian) = cv2.projectPoints(np.array([(-350.0, 270, 0.0)]), rotation_vector, translation_vector,
                                               camera_matrix, dist_coeffs)
            (b4, jacobian) = cv2.projectPoints(np.array([(350.0, -270.0, 0.0)]), rotation_vector,
                                               translation_vector, camera_matrix, dist_coeffs)

            (b11, jacobian) = cv2.projectPoints(np.array([(450.0, 350.0, 400.0)]), rotation_vector,
                                                translation_vector, camera_matrix, dist_coeffs)
            (b12, jacobian) = cv2.projectPoints(np.array([(-450.0, -350.0, 400.0)]), rotation_vector,
                                                translation_vector, camera_matrix, dist_coeffs)
            (b13, jacobian) = cv2.projectPoints(np.array([(-450.0, 350, 400.0)]), rotation_vector,
                                                translation_vector, camera_matrix, dist_coeffs)
            (b14, jacobian) = cv2.projectPoints(np.array([(450.0, -350.0, 400.0)]), rotation_vector,
                                                translation_vector, camera_matrix, dist_coeffs)

            b1 = (int(b1[0][0][0]), int(b1[0][0][1]))
            b2 = (int(b2[0][0][0]), int(b2[0][0][1]))
            b3 = (int(b3[0][0][0]), int(b3[0][0][1]))
            b4 = (int(b4[0][0][0]), int(b4[0][0][1]))

            b11 = (int(b11[0][0][0]), int(b11[0][0][1]))
            b12 = (int(b12[0][0][0]), int(b12[0][0][1]))
            b13 = (int(b13[0][0][0]), int(b13[0][0][1]))
            b14 = (int(b14[0][0][0]), int(b14[0][0][1]))

            if draw_rect1 == True:
                cv2.line(frame, b1, b3, (255, 255, 0), 10)
                cv2.line(frame, b3, b2, (255, 255, 0), 10)
                cv2.line(frame, b2, b4, (255, 255, 0), 10)
                cv2.line(frame, b4, b1, (255, 255, 0), 10)

            if draw_rect2 == True:
                cv2.line(frame, b11, b13, (255, 255, 0), 10)
                cv2.line(frame, b13, b12, (255, 255, 0), 10)
                cv2.line(frame, b12, b14, (255, 255, 0), 10)
                cv2.line(frame, b14, b11, (255, 255, 0), 10)

            if draw_lines == True:
                cv2.line(frame, b11, b1, (0, 255, 0), 10)
                cv2.line(frame, b13, b3, (0, 255, 0), 10)
                cv2.line(frame, b12, b2, (0, 255, 0), 10)
                cv2.line(frame, b14, b4, (0, 255, 0), 10)

        return frame


face_orientation_obj = FaceOrientation()




def Dlib(Image):
    detector = dlib.get_frontal_face_detector()
    predictor_path = "D:\\image\\shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    gray_image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
    # 人脸检测
    faces = detector(gray_image)

    # 68个关键点预测
    for face in faces:
        # 获取68个关键点
        shape = predictor(gray_image, face)
        # 关键点数组
        landmarks = []
        # 将关键点绘制到图像上
        for i in range(68):
            x = shape.part(i).x
            y = shape.part(i).y
            cv2.circle(Image, (x, y), 2, (0, 255, 0), -1)
    return Image


def torchlm_1(image_1):
    torchlm.runtime.bind(faceboxesv2(device="cpu"))
    # 检测模型
    torchlm.runtime.bind(
        pipnet(backbone="resnet18", pretrained=True,
               num_nb=10, num_lms=68, net_stride=32, input_size=256,
               meanface_type="300w", map_location="cpu", checkpoint=None)
    )

    # 读取图像
    # 68个关键点以及脸部检测
    landmarks, bboxes = torchlm.runtime.forward(image_1)

    # 绘制人脸检测框
    # image = torchlm.utils.draw_bboxes(image, bboxes=bboxes)
    # 绘制68个关键点
    Image = torchlm.utils.draw_landmarks(image_1, landmarks=landmarks, color=(255, 0, 0))
    return Image


def line(image):
    torchlm.runtime.bind(
        pipnet(backbone="resnet18", pretrained=True,
               num_nb=10, num_lms=68, net_stride=32, input_size=256,
               meanface_type="300w", map_location="cpu", checkpoint=None)
    )
    # 68个关键点以及脸部检测
    landmarks, bboxes = torchlm.runtime.forward(image)

    # 绘制人脸检测框
    # image = torchlm.utils.draw_bboxes(image, bboxes=bboxes)
    # 绘制68个关键点
    image = torchlm.utils.draw_landmarks(image, landmarks=landmarks, color=(255, 0, 0))

    # 68个关键点数组
    Landmarks = []

    for i in range(landmarks.shape[0]):
        for j in range(landmarks[i].shape[0]):
            x, y = landmarks[i, j, :].astype(int).tolist()
            Landmarks.append((x, y))

    # 绘制关键点连线
    for i in range(0, 16):
        cv2.line(image, Landmarks[i], Landmarks[i + 1], (255, 0, 0), thickness=1)
    for i in range(17, 21):
        cv2.line(image, Landmarks[i], Landmarks[i + 1], (255, 0, 0), thickness=1)
    for i in range(22, 26):
        cv2.line(image, Landmarks[i], Landmarks[i + 1], (255, 0, 0), thickness=1)
    for i in range(27, 35):
        cv2.line(image, Landmarks[i], Landmarks[i + 1], (255, 0, 0), thickness=1)
    for i in range(36, 41):
        cv2.line(image, Landmarks[i], Landmarks[i + 1], (255, 0, 0), thickness=1)
    for i in range(42, 47):
        cv2.line(image, Landmarks[i], Landmarks[i + 1], (255, 0, 0), thickness=1)
    for i in range(48, 67):
        cv2.line(image, Landmarks[i], Landmarks[i + 1], (255, 0, 0), thickness=1)

    detector = dlib.get_frontal_face_detector()
    predictor_path = "D:\\image\\shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 人脸检测
    faces = detector(gray_image)
    # 68个关键点预测
    for face in faces:
    # 获取68个关键点
        shape = predictor(gray_image, face)
        # 关键点数组
        landmarks = []
        # 将关键点绘制到图像上
        for i in range(68):
            x = shape.part(i).x
            y = shape.part(i).y
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            landmarks.append((x, y))
        for i in range(0, 16):
            cv2.line(image, landmarks[i], landmarks[i+1] , (0, 255, 0), thickness=1)
        # 连接左眉毛关键点
        for i in range(17, 21):
            cv2.line(image, landmarks[i], landmarks[i+1], (0, 255, 0), thickness=1)
        # 连接右眉毛关键点
        for i in range(22, 26):
            cv2.line(image, landmarks[i], landmarks[i+1], (0, 255, 0), thickness=1)
        # 连接鼻部关键点
        for i in range(27, 35):
            cv2.line(image, landmarks[i], landmarks[i+1], (0, 255, 0), thickness=1)
        # 连接左眼关键点
        for i in range(36, 41):
            cv2.line(image, landmarks[i], landmarks[i+1], (0, 255, 0), thickness=1)
        # 连接右眼关键点
        for i in range(42, 47):
            cv2.line(image, landmarks[i], landmarks[i+1 ], (0, 255, 0), thickness=1)
        # 连接嘴部关键点
        for i in range(48, 67):
            cv2.line(image, landmarks[i], landmarks[i+1], (0, 255, 0), thickness=1)

        return image



class FaceProcessing(object):
    def __init__(self, ui_obj):
        self.name = "Face Image Processing"
        self.description = "Call for Face Image and video Processing"
        self.ui_obj = ui_obj

    def take_webcam_photo(self, image):
        return image#直接返回图像

    def take_webcam_video(self, images):
        return images#直接返回图像->视频

    def mp_webcam_photo(self, image):
        return image#直接返回图像

    def mp_webcam_face_mesh(self, image):#脸部网格
        mesh_image = apply_media_pipe_facemesh(image)#调用apply函数输出运行后的图像
        return mesh_image

    def mp_webcam_face_detection(self, image):#脸部检测
        face_detection_img = apply_media_pipe_face_detection(image)
        return face_detection_img

    def dlib_apply_face_orientation(self, image):#脸部方向
        image = face_orientation_obj.create_orientation(image)
        return image

    def webcam_stream_update(self, video_frame):
        video_out = face_orientation_obj.create_orientation(video_frame)
        return video_out

    def dlib(self, image):
        dlib_image = Dlib(image)
        return dlib_image

    def torchlm_1(self,image):
        torch_image = torchlm_1(image)
        return torch_image

    def line(self,image):
        line_image = line(image)
        return line_image

    #ui界面
    def create_ui(self):
        with self.ui_obj:
            gr.Markdown("Face Analysis with Webcam/Video")
            with gr.Tabs():
                with gr.TabItem("Playing with Webcam"):#标签1名字
                    with gr.Row():
                        webcam_image_in = gr.Image(label="Webcam Image Input" )
                        webcam_video_in = gr.Video(label="Webcam Video Input" )
                        #框上文字
                    with gr.Row():
                        webcam_photo_action = gr.Button("Take the Photo")
                        webcam_video_action = gr.Button("Take the Video")
                        #按钮文字
                    with gr.Row():
                        webcam_photo_out = gr.Image(label="Webcam Photo Output")
                        webcam_video_out = gr.Video(label="Webcam Video")


                with gr.TabItem("Mediapipe Facemesh with Webcam"):#标签2名字
                    with gr.Row():
                        with gr.Column():
                            mp_image_in = gr.Image(label="Webcam Image Input" )
                        with gr.Column():
                            #设置三个按钮
                            mp_photo_action = gr.Button("Take the Photo")
                            mp_apply_fm_action = gr.Button("Apply Face Mesh the Photo")
                            mp_apply_landmarks_action = gr.Button("Apply Face Landmarks the Photo")
                    with gr.Row():
                        mp_photo_out = gr.Image(label="Webcam Photo Output")
                        mp_fm_photo_out = gr.Image(label="Face Mesh Photo Output")
                        mp_lm_photo_out = gr.Image(label="Face Landmarks Photo Output")


                with gr.TabItem("DLib Based Face Orientation"):#标签3名字
                    with gr.Row():
                        with gr.Column():
                            dlib_image_in = gr.Image(label="Webcam Image Input" )
                        with gr.Column():
                            dlib_photo_action = gr.Button("Take the Photo")
                            dlib_apply_orientation_action = gr.Button("Apply Face Mesh the Photo")
                    with gr.Row():
                        dlib_photo_out = gr.Image(label="Webcam Photo Output")
                        dlib_orientation_photo_out = gr.Image(label="Face Mesh Photo Output")#标签1名字


                with gr.TabItem("Face Orientation on Live Webcam Stream"):#标签4名字
                    with gr.Row():
                        webcam_stream_in = gr.Image(label="Webcam Stream Input",
                                                    streaming=True)
                        webcam_stream_out = gr.Image(label="Webcam Stream Output")
                        webcam_stream_in.change(
                            self.webcam_stream_update,
                            inputs=webcam_stream_in,
                            outputs=webcam_stream_out
                        )


                #新添加的模块，名为models comparison
                with gr.TabItem("models comparison"):#标签5名字
                    with gr.Row():

                        with gr.Column():
                            lib_image_in = gr.Image(label="Webcam Image Input" )

                        with gr.Column():

                            photo_action = gr.Button("Take the Photo")
                            dlib_1 = gr.Button("dlib")
                            torchlm_1 = gr.Button("torchlm")
                            line = gr.Button("line")


                    with gr.Row():
                        lib_photo_out = gr.Image(label="Webcam Photo Output")
                        dlib_out = gr.Image(label="dlib")
                        torchlm_out = gr.Image(label="torchlm")
                        line_out = gr.Image(label="line")



            #click配置

            dlib_photo_action.click(
                self.mp_webcam_photo,
                [
                    dlib_image_in
                ],
                [
                    dlib_photo_out
                ]
            )
            #使用Dlib库来处理dlib_image_in（输入图像），并将处理结果输出到dlib_photo_out
            dlib_apply_orientation_action.click(
                self.dlib_apply_face_orientation,#脸部检测
                [
                    dlib_image_in
                ],
                [
                    dlib_orientation_photo_out
                ]
            )
            #应用面部方向的调整到输入图像dlib_image_in上，并将结果输出到dlib_orientation_photo_out
            mp_photo_action.click(
                self.mp_webcam_photo,
                [
                    mp_image_in
                ],
                [
                    mp_photo_out
                ]
            )
            mp_apply_fm_action.click(
                self.mp_webcam_face_mesh,
                [
                    mp_image_in
                ],
                [
                    mp_fm_photo_out
                ]
            )
            mp_apply_landmarks_action.click(
                self.mp_webcam_face_detection,
                [
                    mp_image_in
                ],
                [
                    mp_lm_photo_out
                ]
            )
            webcam_photo_action.click(
                self.take_webcam_photo,
                [
                    webcam_image_in
                ],
                [
                    webcam_photo_out
                ]
            )
#添加一个新的按键功能：
            photo_action.click(
                self.mp_webcam_photo,
                [
                    lib_image_in
                ],
                [
                    lib_photo_out
                ]

            )
            dlib_1.click(
                self.dlib,
                [
                    lib_image_in
                ],
                [
                    dlib_out
                ]
    )

            torchlm_1.click(
                self.torchlm_1,
                [
                    lib_image_in
                ],
                [
                    torchlm_out
                ]
            )

            line.click(
                self.line,
                [
                    lib_image_in
                ],
                [
                    line_out
                ]
            )



    def launch_ui(self):
        self.ui_obj.launch()


if __name__ == '__main__':
    my_app = gr.Blocks()
    face_ui = FaceProcessing(my_app)
    face_ui.create_ui()
    face_ui.launch_ui()

