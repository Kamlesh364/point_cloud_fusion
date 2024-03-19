#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
# from point_cloud_fusion.utils.point_cloud2 import read_points, create_cloud
import numpy as np
from point_cloud_fusion.utils import ros2_numpy
from cv_bridge import CvBridge, CvBridgeError
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv2 import circle, imshow, waitKey
import tf2_ros
from .utils.image_geometry import PinholeCameraModel
from .utils.geometry import quaternion_matrix
import PyKDL
from tf2_ros import TransformException

CAMERA_MODEL = PinholeCameraModel()

class PointCloudFusionNode(Node):
    def __init__(self, camera_info_topic='/zed/zed_node/rgb/camera_info', 
                        camera_topic='/zed/zed_node/rgb/image_rect_color', 
                        lidar_topic='/velodyne_points', 
                        camera_lidar_topic='/fused_points'):
        
        super().__init__('point_cloud_fusion_node')
        self.get_logger().info('Starting PointCloudFusionNode')

        # Declare parameters
        camera_info_topic = self.get_parameter_or('camera_info_topic', '/zed/zed_node/rgb/camera_info')
        camera_topic = self.get_parameter_or('camera_topic', '/zed/zed_node/rgb/image_rect_color')
        lidar_topic = self.get_parameter_or('lidar_topic', '/velodyne_points')
        camera_lidar_topic = self.get_parameter_or('camera_lidar_topic', '/fused_points')

        # print parameters
        self.get_logger().info('camera_info_topic: {}'.format(camera_info_topic))
        self.get_logger().info('image_topic: {}'.format(camera_topic))
        self.get_logger().info('lidar_topic: {}'.format(lidar_topic))
        self.get_logger().info('camera_lidar_topic: {}'.format(camera_lidar_topic))


        # Calibration data
        self.T = np.array([-0.11314803, -0.03314803, 0.04614803]) # Translation
        self.euler_angles = np.array([0, 0, 0.0200]) # Rotation, to calculate quaternions
        self.camera_matrix = np.array([[395.17385, 0., 347.99444],
                                        [0., 402.15457, 172.23615],
                                        [0., 0., 1.]])

        # Set the topics and callbacks
        self.camera_info = Subscriber(
            self,
            CameraInfo,
            camera_info_topic,
            # self.camera_info_callback
        )

        self.subscription_lidar = Subscriber(
            self,
            PointCloud2,
            lidar_topic,
            # self.lidar_callback
        )

        self.subscription_camera = Subscriber(
            self,
            Image,
            camera_topic,
            # self.image_callback
        )

        # Set the publisher
        self.publisher_fused = self.create_publisher(Image, camera_lidar_topic, 100)
        
        # Get the parameters
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # print the information about the messages subscribed and target frame
        self.get_logger().info('LiDAR Subscribed to: {}'.format(lidar_topic))
        self.get_logger().info('Camera Image Subscribed to: {}'.format(camera_topic))
        self.get_logger().info('Camera Info Subscribed to: {}'.format(camera_info_topic))
        self.get_logger().info('Fused Point Cloud Published to: {}'.format(camera_lidar_topic))

        self.ts = ApproximateTimeSynchronizer([self.subscription_lidar, self.subscription_camera, self.camera_info], 100, 0.1)
        self.ts.registerCallback(self.callback)
        self.get_logger().info('TimeSynchronizer created.')
    
    def callback(self, lidar_msg, camera_image_msg, camera_info_msg):
        global CAMERA_MODEL
        CAMERA_MODEL.fromCameraInfo(camera_info_msg)
        try:
            # Read image using CV bridge
            if hasattr(camera_image_msg, 'encoding'):
                try:
                    img = CvBridge().imgmsg_to_cv2(camera_image_msg, desired_encoding='bgr8')
                except CvBridgeError as e: 
                    self.get_logger().error('Image acquisition error: {}'.format(e))
                    return
            else:
                self.get_logger().error('No encoding found in the image message!')
                return
            
            transform = self.tf_buffer.lookup_transform(
                                    'zed_camera_center', 
                                    lidar_msg.header.frame_id, 
                                    lidar_msg.header.stamp)
        except TransformException as ex:
            self.get_logger().error('TransformException: {}'.format(ex))
            return
        
        translation = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])
        translation = translation + (1,)
        
        Rq = quaternion_matrix(np.array([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]))
        Rq[0, 3] = transform.transform.translation.x
        Rq[1, 3] = transform.transform.translation.y
        Rq[2, 3] = transform.transform.translation.z

        # Draw Lidar points on the camera image
        if lidar_msg is not None and CAMERA_MODEL is not None:
            projected_points = self.project_lidar_points(img, CAMERA_MODEL, lidar_msg)
            self.get_logger().info("Projected Points: {}".format(projected_points))
            for point in projected_points:
                if np.isnan(point).any():
                    self.get_logger().info('Skipping point with NaN values.')
                    continue  # Skip this point if it contains NaN values
                center_x, center_y = int(point[0]), int(point[1])
                try:
                    circle(img, (center_x, center_y), radius=2, color=(0, 0, 255), thickness=-1)
                    imshow("projected", img)
                    waitKey(1)
                except Exception as e:
                    self.get_logger().error('Exception: {}'.format(e))
                    return
        try:
            # Publish the fused point cloud
            img_msg = CvBridge().cv2_to_imgmsg(img, encoding='bgr8')
            self.publisher_fused.publish(img_msg)
        except CvBridgeError as e:
            self.get_logger().error('CvBridgeError: {}'.format(e))
            return

    def project_lidar_points(self, image, camera_model, lidar_msg):
        # Get the image size
        image_size = (image.shape[1], image.shape[0])

        # Convert Lidar message to numpy array
        points3D = ros2_numpy.pointcloud2_to_array(lidar_msg)

        # Initialize an empty list to store projected points from all channels
        all_projected_points = []

        # Process each channel separately
        for channel_index in range(16):
            # Extract x, y, and z components from the Lidar points array for the current channel
            x = points3D['x'][:, channel_index]
            y = points3D['y'][:, channel_index]
            z = points3D['z'][:, channel_index]

            # Convert x, y, and z components to np.float32 and stack them together
            points3D_float32 = np.column_stack((x, y, z)).astype(np.float32)

            # Project Lidar points onto the camera image using camera intrinsic parameters
            projected_points_homogeneous = np.dot(camera_model.K, points3D_float32.T)
            projected_points_homogeneous /= projected_points_homogeneous[2]  # Divide by z-coordinate

            # Extract (x, y) coordinates from homogeneous coordinates
            projected_points = projected_points_homogeneous[:2].T

            # Clip points that fall outside of the image bounds
            projected_points[:, 0] = np.clip(projected_points[:, 0], 0, image_size[1])
            projected_points[:, 1] = np.clip(projected_points[:, 1], 0, image_size[0])

            # Append the projected points for the current channel to the list
            all_projected_points.append(projected_points)

        # Combine the projected points from all channels
        all_projected_points = np.concatenate(all_projected_points, axis=1)

        return all_projected_points
    

def main(args=None):
    rclpy.init(args=args)

    node = PointCloudFusionNode()

    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
