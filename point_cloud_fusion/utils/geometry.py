from .registry import converts_from_numpy, converts_to_numpy
from geometry_msgs.msg import Transform, Vector3, Quaternion, Point, Pose
import tf2_ros as transformations
from . import numpify
from math import sqrt
import numpy as np
import math

# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

# basic types

@converts_to_numpy(Vector3)
def vector3_to_numpy(msg, hom=False):
    if hom:
        return np.array([msg.x, msg.y, msg.z, 0])
    else:
        return np.array([msg.x, msg.y, msg.z])

@converts_from_numpy(Vector3)
def numpy_to_vector3(arr):
    if arr.shape[-1] == 4:
        assert np.all(arr[...,-1] == 0)
        arr = arr[...,:-1]

    if len(arr.shape) == 1:
        return Vector3(**dict(zip(['x', 'y', 'z'], arr)))
    else:
        return np.apply_along_axis(
            lambda v: Vector3(**dict(zip(['x', 'y', 'z'], v))), axis=-1,
            arr=arr)

@converts_to_numpy(Point)
def point_to_numpy(msg, hom=False):
    if hom:
        return np.array([msg.x, msg.y, msg.z, 1])
    else:
        return np.array([msg.x, msg.y, msg.z])

@converts_from_numpy(Point)
def numpy_to_point(arr):
    if arr.shape[-1] == 4:
        arr = arr[...,:-1] / arr[...,-1]

    if len(arr.shape) == 1:
        return Point(**dict(zip(['x', 'y', 'z'], arr)))
    else:
        return np.apply_along_axis(
            lambda v: Point(**dict(zip(['x', 'y', 'z'], v))), axis=-1, arr=arr)

@converts_to_numpy(Quaternion)
def quat_to_numpy(msg):
    return np.array([msg.x, msg.y, msg.z, msg.w])

@converts_from_numpy(Quaternion)
def numpy_to_quat(arr):
    assert arr.shape[-1] == 4

    if len(arr.shape) == 1:
        return Quaternion(**dict(zip(['x', 'y', 'z', 'w'], arr)))
    else:
        return np.apply_along_axis(
            lambda v: Quaternion(**dict(zip(['x', 'y', 'z', 'w'], v))),
            axis=-1, arr=arr)


# compound types
# all of these take ...x4x4 homogeneous matrices

@converts_to_numpy(Transform)
def transform_to_numpy(msg):
    return np.dot(
        transformations.translation_matrix(numpify(msg.translation)),
        transformations.quaternion_matrix(numpify(msg.rotation))
    )

@converts_from_numpy(Transform)
def numpy_to_transform(arr):
    shape, rest = arr.shape[:-2], arr.shape[-2:]
    assert rest == (4,4)

    if len(shape) == 0:
        trans = transformations.translation_from_matrix(arr)
        quat = transformations.quaternion_from_matrix(arr)

        return Transform(
            translation=Vector3(**dict(zip(['x', 'y', 'z'], trans))),
            rotation=Quaternion(**dict(zip(['x', 'y', 'z', 'w'], quat)))
        )
    else:
        res = np.empty(shape, dtype=np.object_)
        for idx in np.ndindex(shape):
            res[idx] = Transform(
                translation=Vector3(
                    **dict(
                        zip(['x', 'y', 'z'],
                        transformations.translation_from_matrix(arr[idx])))),
                rotation=Quaternion(
                    **dict(
                        zip(['x', 'y', 'z', 'w'],
                        transformations.quaternion_from_matrix(arr[idx]))))
            )

@converts_to_numpy(Pose)
def pose_to_numpy(msg):
    return np.dot(
        transformations.translation_matrix(numpify(msg.position)),
        transformations.quaternion_matrix(numpify(msg.orientation))
    )

@converts_from_numpy(Pose)
def numpy_to_pose(arr):
    shape, rest = arr.shape[:-2], arr.shape[-2:]
    assert rest == (4,4)

    if len(shape) == 0:
        trans = transformations.translation_from_matrix(arr)
        quat = transformations.quaternion_from_matrix(arr)

        return Pose(
            position=Point(**dict(zip(['x', 'y', 'z'], trans))),
            orientation=Quaternion(**dict(zip(['x', 'y', 'z', 'w'], quat)))
        )
    else:
        res = np.empty(shape, dtype=np.object_)
        for idx in np.ndindex(shape):
            res[idx] = Pose(
                position=Point(
                    **dict(
                        zip(['x', 'y', 'z'],
                        transformations.translation_from_matrix(arr[idx])))),
                orientation=Quaternion(
                    **dict(
                        zip(['x', 'y', 'z', 'w'],
                        transformations.quaternion_from_matrix(arr[idx]))))
            )

@converts_to_numpy(np.ndarray)
def quaternion_from_matrix(matrix):
    """Return quaternion from rotation matrix.

    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.0164262, 0.0328524, 0.0492786, 0.9981095])
    True

    """
    q = np.empty((4, ), dtype=np.float64)
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    t = np.trace(M)
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    q *= 0.5 / sqrt(t * M[3, 3])
    return q

def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    True

    """
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)

def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
    """Return quaternion from Euler angles and axis sequence.

    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple

    >>> q = quaternion_from_euler(1, 2, 3, 'ryxz')
    >>> numpy.allclose(q, [0.310622, -0.718287, 0.444435, 0.435953])
    True

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    quaternion = np.empty((4, ), dtype=np.float64)
    if repetition:
        quaternion[i] = cj*(cs + sc)
        quaternion[j] = sj*(cc + ss)
        quaternion[k] = sj*(cs - sc)
        quaternion[3] = cj*(cc - ss)
    else:
        quaternion[i] = cj*sc - sj*cs
        quaternion[j] = cj*ss + sj*cc
        quaternion[k] = cj*cs - sj*sc
        quaternion[3] = cj*cc + sj*ss
    if parity:
        quaternion[j] *= -1

    return quaternion