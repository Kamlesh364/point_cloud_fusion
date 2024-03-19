from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription #, ExecuteProcess
# from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource

from os.path import join


def generate_launch_description():
    # Setup params for Camera-LiDAR calibration script
    # camera_info_topic = DeclareLaunchArgument('camera_info_topic', default_value='/zed/zed_node/rgb/camera_info')
    # camera_topic = DeclareLaunchArgument('image_color_topic', default_value='/zed/zed_node/rgb/image_rect_color')
    # lidar_topic = DeclareLaunchArgument('velodyne_points_topic', default_value='/velodyne_points')
    # camera_lidar_topic = DeclareLaunchArgument('camera_lidar_topic', default_value='/fused_points')

    # camera_bag = DeclareLaunchArgument('camera_bag', default_value='/workspaces/isaac_ros-dev/ros2_camera/')
    # LiDAR_bag = DeclareLaunchArgument('LiDAR_bag', default_value='/workspaces/isaac_ros-dev/ros1_lidar/')

    Launch_args = [
        DeclareLaunchArgument('camera_info_topic', default_value='/zed/zed_node/rgb/camera_info'),
        DeclareLaunchArgument('camera_topic', default_value='/zed/zed_node/rgb/image_rect_color'),
        DeclareLaunchArgument('velodyne_points_topic', default_value='/velodyne_points'),
        DeclareLaunchArgument('lidar_topic', default_value='/fused_points'),
        DeclareLaunchArgument('camera_bag', default_value='/workspaces/isaac_ros-dev/camera_checkboard/'),
        DeclareLaunchArgument('LiDAR_bag', default_value='/workspaces/isaac_ros-dev/lidar_checkboard/'),
    ]

    # camera_bag = LaunchConfiguration('camera_bag')
    # LiDAR_bag = LaunchConfiguration('LiDAR_bag')
    # camera_info_topic = LaunchConfiguration('camera_info_topic')
    # camera_topic = LaunchConfiguration('camera_topic')

    # rosbag_p1 = ExecuteProcess(cmd=['ros2', 'bag', 'play', camera_bag], output='screen', log_cmd=False)
    # rosbag_p2 = ExecuteProcess(cmd=['ros2', 'bag', 'play', LiDAR_bag], output='screen', log_cmd=False)

    tf_static_lidar = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_velodyne_tf',
        output='screen',
        arguments=['-0.11314803', '-0.03314803', '0.04614803', '0', '0', '0.0200', 'zed_camera_center', 'velodyne']
    )

    image_proc = Node(
        package='image_proc',
        executable='image_proc',
        name='image_proc_node1'
    )

    fusion_node = Node(
        package='point_cloud_fusion',
        executable='lidar_image',
        name='lidar_image_node',
        output='log',
        log_cmd=True
    )

    velodyne_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(join(get_package_share_directory('velodyne'),'launch/velodyne-all-nodes-VLP16-launch.py')),
    )

    zed_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(join(get_package_share_directory('zed_wrapper'),'launch/obsolete/zed.launch.py'))
    )

    final_launch_description = LaunchDescription([
        *Launch_args,
        velodyne_launch,
        zed_launch,
        # rosbag_p1,
        # rosbag_p2,
        tf_static_lidar,
        # tf_static_camera,
        # nodelet_manager,
        image_proc,
        # image_view,
        fusion_node
    ])

    return final_launch_description