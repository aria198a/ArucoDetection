#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
import cv2
import numpy as np


class ArucoUsbCamNode(Node):
    def __init__(self):
        super().__init__('aruco_usb_cam_node')

        # ---- 參數 ----
        self.declare_parameter('video_device', '/dev/video0')
        self.declare_parameter('marker_length', 0.05)  # 5cm
        self.declare_parameter('detection_rate', 30.0)

        # camera intrinsics for 1280x720（之後可以用標定值覆蓋）
        self.declare_parameter('camera.fx', 900.0)
        self.declare_parameter('camera.fy', 900.0)
        self.declare_parameter('camera.cx', 640.0)  # width/2
        self.declare_parameter('camera.cy', 360.0)  # height/2

        # ---- 讀參數 ----
        self.video_device = self.get_parameter('video_device').get_parameter_value().string_value
        self.marker_length = self.get_parameter('marker_length').get_parameter_value().double_value
        self.detection_rate = self.get_parameter('detection_rate').get_parameter_value().double_value

        fx = self.get_parameter('camera.fx').get_parameter_value().double_value
        fy = self.get_parameter('camera.fy').get_parameter_value().double_value
        cx = self.get_parameter('camera.cx').get_parameter_value().double_value
        cy = self.get_parameter('camera.cy').get_parameter_value().double_value

        # 720P
        self.width = 1280
        self.height = 720

        # 相機內參矩陣 & 畸變（預設 0）
        self.camera_matrix = np.array(
            [[fx, 0.0, cx],
             [0.0, fy, cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float32
        )
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        # ---- 初始化 USB 相機 ----
        self.cap = cv2.VideoCapture(self.video_device)
        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open video device: {self.video_device}")
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.detection_rate)
            self.get_logger().info(
                f"Opened {self.video_device} with resolution {self.width}x{self.height} @ {self.detection_rate} FPS"
            )

        # ---- ArUco ----
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

        # 兼容舊版 / 新版 OpenCV API
        if hasattr(cv2.aruco, 'DetectorParameters_create'):
            # 舊版 API
            self.get_logger().info("Using old ArUco API (DetectorParameters_create + detectMarkers function)")
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.use_new_api = False
            self.detector = None
        else:
            # 新版 API
            self.get_logger().info("Using new ArUco API (DetectorParameters + ArucoDetector)")
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.use_new_api = True

        # ---- Publisher ----
        self.pose_pub = self.create_publisher(PoseStamped, 'aruco_pose', 10)

        # ---- Timer ----
        timer_period = 1.0 / self.detection_rate
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # log 節流用（nanoseconds）
        self.last_log_time_ns = 0

        self.get_logger().info("ArucoUsbCamNode started (no cv_bridge / image output).")

    def timer_callback(self):
        if self.cap is None or (not self.cap.isOpened()):
            self.get_logger().warn_once("VideoCapture not opened.")
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.get_logger().warn("Failed to read frame from camera.")
            return

        frame = cv2.resize(frame, (self.width, self.height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---- 偵測 ArUco ----
        if self.use_new_api:
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray,
                self.aruco_dict,
                parameters=self.aruco_params
            )

        if ids is not None and len(ids) > 0:
            # 在畫面上畫出偵測到的標記（純 debug）
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # 這裡示範只處理第一個 marker
            first_corner = corners[0]
            marker_id = int(ids[0][0])

            rvec, tvec = self.estimate_pose_single_marker(first_corner, self.marker_length)
            if rvec is None or tvec is None:
                return  # 這一幀姿態計算失敗就略過

            qx, qy, qz, qw = self.rvec_to_quaternion(rvec)

            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "usb_camera"

            pose_msg.pose.position.x = float(tvec[0])
            pose_msg.pose.position.y = float(tvec[1])
            pose_msg.pose.position.z = float(tvec[2])
            pose_msg.pose.orientation.x = float(qx)
            pose_msg.pose.orientation.y = float(qy)
            pose_msg.pose.orientation.z = float(qz)
            pose_msg.pose.orientation.w = float(qw)

            self.pose_pub.publish(pose_msg)

            # 手動做 1 秒鐘節流的 log
            now_ns = self.get_clock().now().nanoseconds
            if now_ns - self.last_log_time_ns > 1_000_000_000:  # 1 秒
                self.get_logger().info(
                    f"Detected marker ID={marker_id}, "
                    f"t=({tvec[0]:.3f},{tvec[1]:.3f},{tvec[2]:.3f}) m"
                )
                self.last_log_time_ns = now_ns

        # 顯示畫面（如果有桌面環境）
        cv2.imshow("Aruco USB Cam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info("Quit by user (pressed q).")
            # 這裡直接 shutdown ROS
            rclpy.shutdown()

    def estimate_pose_single_marker(self, corner, marker_length):
        """
        使用 solvePnP 自己算單一 ArUco marker 的 rvec, tvec
        corner: 單一 marker 的 corners, 可能形狀是 (1,4,2) 或 (4,1,2) 或 (4,2)
        marker_length: 邊長 (m)
        """
        img_pts = np.array(corner, dtype=np.float32).reshape(-1, 2)
        if img_pts.shape[0] != 4:
            self.get_logger().warn(f"Expected 4 corners, got {img_pts.shape[0]}")
            return None, None

        half = marker_length / 2.0

        # 定義 marker 在自身座標系的 3D 角點 (順序要跟 img_pts 對應)
        # 假設順序為: [top-left, top-right, bottom-right, bottom-left]
        obj_pts = np.array([
            [-half,  half, 0.0],
            [ half,  half, 0.0],
            [ half, -half, 0.0],
            [-half, -half, 0.0],
        ], dtype=np.float32)

        # 某些 OpenCV 可能沒有 SOLVEPNP_IPPE_SQUARE，就 fallback 到 ITERATIVE
        pnp_flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)

        ok, rvec, tvec = cv2.solvePnP(
            obj_pts,
            img_pts,
            self.camera_matrix,
            self.dist_coeffs,
            flags=pnp_flag
        )

        if not ok:
            self.get_logger().warn("solvePnP failed for marker.")
            return None, None

        return rvec.reshape(3), tvec.reshape(3)

    def rvec_to_quaternion(self, rvec):
        """rvec (3,) -> (qx,qy,qz,qw)"""
        R, _ = cv2.Rodrigues(rvec)
        tr = R[0, 0] + R[1, 1] + R[2, 2]

        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2.0
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

        return qx, qy, qz, qw

    def destroy_node(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ArucoUsbCamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
