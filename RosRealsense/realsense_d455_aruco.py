import cv2
from cv2 import aruco
import numpy as np
import pyrealsense2 as rs


class MarkerDetectionSystem:
    def __init__(self, marker_size_cm, camera_matrix, dist_coeffs):
        """
        marker_size_cm : 你列印出來的 ArUco 邊長（公分）
        camera_matrix  : 相機內參 (3x3)
        dist_coeffs    : 畸變參數 (1x5)
        """
        self.marker_size_cm = marker_size_cm
        # 轉成「公尺」，讓 tvec 單位是公尺，比較正常
        self.marker_size_m = marker_size_cm / 100.0

        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        self.detected_markers = {}
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def detect_markers(self, frame):
        """
        偵測畫面中的 ArUco，並存：
        - ID
        - 質心 (cX, cY)
        - 4 個角點
        - rvec, tvec (若有相機內參)
        """
        self.detected_markers = {}  # 每一幀重置

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary)

        if ids is None:
            return self.detected_markers

        # 如果有內參，直接一次估計所有 marker 的姿態
        rvecs, tvecs = None, None
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size_m, self.camera_matrix, self.dist_coeffs
            )

        for i in range(len(ids)):
            # 算質心
            M = cv2.moments(corners[i][0])
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            marker_info = {
                "ID": int(ids[i][0]),
                "Centroid": (cX, cY),
                "Corners": corners[i][0],
            }

            if rvecs is not None and tvecs is not None:
                marker_info["rvec"] = rvecs[i]
                marker_info["tvec"] = tvecs[i]

            self.detected_markers[int(ids[i][0])] = marker_info

        return self.detected_markers

    def draw_markers(self, frame, depth_frame=None):
        """
        在畫面上畫：
        - 質心點
        - 3D 軸（如果有 rvec/tvec）
        - 距離資訊（3D 距離 + 深度距離）
        """
        if not self.detected_markers:
            return frame

        for marker_id, info in self.detected_markers.items():
            cX, cY = info["Centroid"]

            # 畫 marker 質心
            cv2.circle(frame, (cX, cY), 5, (255, 0, 255), -1)

            text_lines = []

            # 1) 使用 tvec 的 3D 距離（公尺 → 公分）
            if "tvec" in info:
                tvec = info["tvec"].reshape(-1)  # (3,)
                dist_cam_m = np.linalg.norm(tvec)
                dist_cam_cm = dist_cam_m * 100.0
                text_lines.append(
                    f"ID {marker_id} | PoseDist: {dist_cam_cm:5.1f} cm"
                )

                # 畫 3D 座標軸（軸長度 0.05 m = 5 cm）
                cv2.drawFrameAxes(
                    frame,
                    self.camera_matrix,
                    self.dist_coeffs,
                    info["rvec"],
                    info["tvec"],
                    0.05
                )


            # 2) 使用 RealSense 深度影像在質心位置讀取距離（公尺）
            if depth_frame is not None:
                # 注意要 clamp 在畫面範圍內
                h, w = frame.shape[:2]
                u = np.clip(cX, 0, w - 1)
                v = np.clip(cY, 0, h - 1)
                depth_m = depth_frame.get_distance(u, v)
                if depth_m > 0:
                    text_lines.append(f"Depth @ center: {depth_m*100:5.1f} cm")

            # 把文字顯示在 marker 質心附近
            for i, line in enumerate(text_lines):
                cv2.putText(
                    frame,
                    line,
                    (cX - 150, cY + 20 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        return frame

    def draw_connections_3d(self, frame):
        """
        額外功能：用 tvec 算不同 marker 之間的 3D 距離（不是必須）
        """
        ids = list(self.detected_markers.keys())
        if len(ids) < 2:
            return frame

        y0 = 30
        for i in range(len(ids) - 1):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                info1, info2 = self.detected_markers[id1], self.detected_markers[id2]
                if "tvec" in info1 and "tvec" in info2:
                    d_m = np.linalg.norm(
                        info1["tvec"].reshape(-1) - info2["tvec"].reshape(-1)
                    )
                    d_cm = d_m * 100.0
                    txt = f"3D dist {id1}-{id2}: {d_cm:5.1f} cm"
                    cv2.putText(
                        frame,
                        txt,
                        (20, y0),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                    y0 += 20

        return frame


def create_realsense_pipeline(width=1280, height=720, fps=30):
    """
    建立並啟動 RealSense 管線（color + depth，並做對齊）
    回傳：pipeline, align, color_intrinsics
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    profile = pipeline.start(config)

    # 對齊 depth 到 color 座標系
    align_to = rs.stream.color
    align = rs.align(align_to)

    # 取得 color stream 的內參
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    # 建立相機內參矩陣
    camera_matrix = np.array(
        [
            [intr.fx, 0, intr.ppx],
            [0, intr.fy, intr.ppy],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    # RealSense 提供的係數，取前 5 個當作 OpenCV 的 k1~k5
    dist_coeffs = np.array(intr.coeffs[:5], dtype=np.float32)

    return pipeline, align, camera_matrix, dist_coeffs


def main():
    # === 1. 啟動 RealSense D455F ===
    pipeline, align, camera_matrix, dist_coeffs = create_realsense_pipeline()

    # 你的 ArUco 實際邊長（單位：cm）
    marker_size_cm = 9.0

    mds = MarkerDetectionSystem(marker_size_cm, camera_matrix, dist_coeffs)

    try:
        while True:
            # 等待一幀對齊後的資料
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # 轉成 numpy 陣列 (BGR)
            color_image = np.asanyarray(color_frame.get_data())

            # 2. 偵測 ArUco
            mds.detect_markers(color_image)

            # 3. 畫 marker、距離、3D 座標軸
            color_image = mds.draw_markers(color_image, depth_frame)
            color_image = mds.draw_connections_3d(color_image)

            # 4. 顯示
            cv2.imshow("D455F ArUco", color_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # q 或 ESC 離開
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
