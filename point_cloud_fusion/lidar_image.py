#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
from .utils.image_geometry import PinholeCameraModel
from .utils.ros2_numpy import quaternion_matrix, pointcloud2_to_xyz_array
import tf2_ros
import numpy as np
import cv2
# import struct
import math
from message_filters import ApproximateTimeSynchronizer, Subscriber

class LidarImage(Node):

    def __init__(self):
        super().__init__('lidar_image')
        
        # self.declare_parameter('image_topic', '/zed/zed_node/rgb/image_rect_color')
        # self.declare_parameter('image_lidar_topic', 'image_lidar')
        # self.declare_parameter('camera_info_topic', 'camera')
        # self.declare_parameter('velodyne_topic', 'velodyne')

        # self._imageInputName = self.get_parameter('image_topic').get_parameter_value().string_value
        # self.get_logger().info(f'Image Input Topic: {self._imageInputName}')

        # self._imageOutputName = self.get_parameter('image_lidar_topic').get_parameter_value().string_value
        # self.get_logger().info(f'Image Output Topic: {self._imageOutputName}')

        # self._cameraName = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        # self.get_logger().info(f'Camera Info Topic: {self._cameraName}')

        # self._velodyneName = self.get_parameter('velodyne_topic').get_parameter_value().string_value
        # self.get_logger().info(f'Velodyne Topic: {self._velodyneName}')

        self._imageInput = Subscriber(self, Image, "/zed/zed_node/rgb/image_rect_color")
        self._camera = Subscriber(self, CameraInfo, "/zed/zed_node/rgb/camera_info")
        self._velodyne = Subscriber(self, PointCloud2, "/velodyne_points")

        self._imageOutput = self.create_publisher(Image, "/fused_image", 10)

        self._bridge = CvBridge()
        self._cameraModel = PinholeCameraModel()
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._velodyneData = None
        self.translation = [-0.11314803, -0.03314803, 0.04614803, 1.0] # Translation
        self.euler_angles = [0, 0, 0.0200] # Rotation, to calculate quaternions
        self.rotation = [0.010000, 0.000000, 0.000000, 0.999950]

        # ApproximateTimeSynchronizer to synchronize the topics
        ats = ApproximateTimeSynchronizer([self._imageInput, self._camera, self._velodyne], queue_size=5, slop=0.1)
        ats.registerCallback(self.callback)

    def callback(self, image_sub, camera_info_sub, velodyne_sub):
        # self.get_logger().info('Received an image!')
        self.cameraCallback(camera_info_sub)
        self.velodyneCallback(velodyne_sub)

        try:
            cv_image = self._bridge.imgmsg_to_cv2(image_sub, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error('Failed to convert image')
            return

        # try:
        # trans = self._tf_buffer.lookup_transform('zed_camera_center', velodyne_sub.header.frame_id, velodyne_sub.header.stamp)
        # translation = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
        # rotation = [transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]

        Rq = quaternion_matrix(self.rotation)
        Rq[:, 3] = self.translation
        # dropped_rot = 0
        # dropped_inten = 0
        # dropped_dist = 0
        # dropped_nan = 0
        if self._velodyneData:
            for i in range(len(self._velodyneData)):
                point = [i+1 for i in self._velodyneData[i]]
                point.append(1.0)
                if not np.isfinite(point).all():
                    # dropped_nan += 1
                    self.get_logger().info('Point is NaN')
                    continue

                if math.sqrt(np.sum(np.array(point[:])**2)) > 10.0:
                    # dropped_dist += 1
                    continue

                # intensity = self._velodyneData[i][3]
                # if intensity < 0.0001:
                #     # dropped_inten += 1
                #     continue

                rotatedPoint = Rq.dot(point)
                if rotatedPoint[2] < 0:
                    # dropped_rot += 1
                    self.get_logger().info('Point is behind the camera')
                    continue
                
                try:
                    uv = self._cameraModel.project3dToPixel(rotatedPoint[:3])

                    if 0 <= uv[0] < image_sub.width and 0 <= uv[1] < image_sub.height:
                        # intensityInt = int(intensity * 255)
                        cv2.circle(cv_image, (int(uv[0]), int(uv[1])), 1, (0, 0, 255), -1)
                except Exception as e:
                    self.get_logger().info('Failed to project point: {}'.format(e))
                    continue

        self._imageOutput.publish(self._bridge.cv2_to_imgmsg(cv_image, 'bgr8'))
        # except Exception as e:
        #     self.get_logger().error('Failed during image processing: {}'.format(e))
        # self.get_logger().info('Dropped {} points due to distance, {} due to intensity, {} due to rotation, {} due to NaN'.format(dropped_dist, dropped_inten, dropped_rot, dropped_nan))

    def cameraCallback(self, data):
        # self.get_logger().info('Received camera info')
        self._cameraModel.fromCameraInfo(data)

    def velodyneCallback(self, data):
        # self.get_logger().info('Received velodyne point cloud in {}.'.format(data.header.frame_id))

        fmt = 'ffff' if data.is_bigendian else '<ffff'
        points3D = pointcloud2_to_xyz_array(data)
        points = points3D.tolist()
        self._velodyneData = points
        # self.get_logger().info('Received {} points'.format(len(self._velodyneData)))


# def velodyneCallback(self, data):
#         # self.get_logger().info('Received velodyne point cloud in {}.'.format(data.header.frame_id))

#         fmt = 'ffff' if data.is_bigendian else '<ffff'
#         points = []

#         for index in range(0, len(data.data), 16):
#             points.append(struct.unpack(fmt, data.data[index:index + 16]))

#         self._velodyneData = points
#         self.get_logger().info('Received {} points'.format(len(self._velodyneData)))

def main(args=None):
    rclpy.init(args=args)
    lidar_image = LidarImage()
    rclpy.spin(lidar_image)
    lidar_image.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
