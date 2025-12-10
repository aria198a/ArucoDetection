import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

import pyrealsense2 as rs
import numpy as np
import cv2
from cv2 import aruco

class ArucoCamNode(Node):
    def __init__(self):
        super().__init__('aruco_cam_node')
        self.pose_pub = self.create_publisher(Float32MultiArray, 'center_pose', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.cam = CamWorker()
        self.get_logger().info("Aruco camera node started.")

    def timer_callback(self):
        color, depth = self.cam.take_frame()
        if color is None:
            return

        markers, vis = self.cam.detect_aruco_with_depth()
        for marker in markers:
            T = marker["T_cam_tag"]
            if T is not None:
                msg = Float32MultiArray()
                msg.data = T.flatten().tolist()
                self.pose_pub.publish(msg)
                self.get_logger().info(f"Published ArUco ID {marker['id']} pose.")

        # 顯示影像
        cv2.imshow("Aruco Depth Measurement", vis)
        cv2.waitKey(1)

    def destroy_node(self):
        super().destroy_node()
        self.cam.stop()

class CamWorker:
    def __init__(self, width=1280, height=720, fps=15):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        self.ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
        self.depth_intrin = None
        self.depth_image = None
        self.color_image = None

    def take_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None
        self.depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        self.depth_image = np.asanyarray(depth_frame.get_data())
        self.color_image = np.asanyarray(color_frame.get_data())
        return self.color_image, self.depth_image

    def pixel_to_camera(self, pixel):
        x = int(np.clip(pixel[0], 0, self.depth_image.shape[1]-1))
        y = int(np.clip(pixel[1], 0, self.depth_image.shape[0]-1))
        depth = self.depth_image[y, x]
        if depth == 0:
            return None
        return np.array(rs.rs2_deproject_pixel_to_point(self.depth_intrin, [x,y], depth))

    def detect_aruco_with_depth(self):
        gray = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)
        params = aruco.DetectorParameters()
        corners, ids, _ = aruco.detectMarkers(gray, self.ARUCO_DICT, parameters=params)
        results = []

        if ids is not None:
            aruco.drawDetectedMarkers(self.color_image, corners, ids)
            for i, marker_id in enumerate(ids.flatten()):
                corners_2d = corners[i][0]
                corners_3d = [self.pixel_to_camera(pt) for pt in corners_2d]
                self.draw_corners_with_3d(self.color_image, corners_2d, corners_3d)
                T, center_3d = self.compute_aruco_pose(corners_3d)

                if T is not None and center_3d is not None:
                    center_px = np.mean(corners_2d, axis=0).astype(int)
                    self.draw_pose_axes(self.color_image, T, tuple(center_px))

                results.append({
                    "id": int(marker_id),
                    "corners_2d": corners_2d,
                    "corners_3d": corners_3d,
                    "center_3d": center_3d,
                    "T_cam_tag": T
                })
        return results, self.color_image

    def draw_corners_with_3d(self, img, corners_2d, corners_3d):
        for i, (pt2d, pt3d) in enumerate(zip(corners_2d, corners_3d)):
            if pt3d is None:
                continue
            x, y = int(pt2d[0]), int(pt2d[1])
            cv2.circle(img, (x, y), 6, (0,0,255), -1)
            cv2.putText(img, f"{i}", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    def compute_aruco_pose(self, corners_3d):
        if any(p is None for p in corners_3d):
            return None, None
        P0, P1, P2, P3 = corners_3d
        center = (P0 + P1 + P2 + P3) / 4.0
        x_axis = P1 - P0
        y_axis = P3 - P0
        x_axis /= np.linalg.norm(x_axis)
        y_axis /= np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        R = np.stack([x_axis, y_axis, z_axis], axis=1)
        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = center
        return T, center

    def draw_pose_axes(self, img, T, origin_pixel, axis_len=50):
        R = T[:3,:3]
        t = T[:3,3]
        colors = [(0,0,255),(0,255,0),(255,0,0)]  # X,Y,Z
        for i in range(3):
            p = t + R[:,i] * axis_len
            p_px = rs.rs2_project_point_to_pixel(self.depth_intrin, p.tolist())
            cv2.line(img, origin_pixel, tuple(map(int, p_px)), colors[i], 2)

    def stop(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = ArucoCamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
