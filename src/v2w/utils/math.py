import torch
    

def quat_to_rot_mat(q: torch.Tensor | Tuple) -> torch.Tensor:
    """
        Converts a quaternion to a rotation matrix.
    Args:
        q (torch.Tensor): A tensor of shape (4,) representing the quaternion [q0, q1, q2, q3].
    Returns:
        torch.Tensor: A tensor of shape (3, 3) representing the rotation matrix.
    """
    
    q0, q1, q2, q3 = q

    R = torch.Tensor([
        [q0**2 + q1**2 - q2**2 - q3**2, 2*q1*q2 - 2*q0*q3, 2*q1*q3 + 2*q0*q2],
        [2*q1*q2 + 2*q0*q3, q0**2 - q1**2 + q2**2 - q3**2, 2*q2*q3 - 2*q0*q1],
        [2*q1*q3 - 2*q0*q2, 2*q2*q3 + 2*q0*q1, q0**2 - q1**2 - q2**2 + q3**2]
        ])

    return R


def rot_mat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """
        Converts a rotation matrixx to a quaternion.
    Args:
        R (torch.Tensor): A tensor of shape (3, 3) representing the rotation matrix.
    Returns:
        q (torch.Tensor): A tensor of shape (4,) representing the quaternion [q0, q1, q2, q3].
    """

    q0_abs = 0.5 * torch.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
    q1_abs = 0.5 * torch.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
    q2_abs = 0.5 * torch.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2])
    q3_abs = 0.5 * torch.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2])

    if q0_abs >= q1_abs and q0_abs >= q2_abs and q0_abs >= q3_abs:
        q0 = q0_abs
        q1 = (R[2, 1] - R[1, 2]) / (4 * q0)
        q2 = (R[0, 2] - R[2, 0]) / (4 * q0)
        q3 = (R[1, 0] - R[0, 1]) / (4 * q0)
    elif q1_abs >= q0_abs and q1_abs >= q2_abs and q1_abs >= q3_abs:
        q1 = q1_abs
        q0 = (R[2, 1] - R[1, 2]) / (4 * q1)
        q2 = (R[0, 1] + R[1, 0]) / (4 * q1)
        q3 = (R[0, 2] + R[2, 0]) / (4 * q1)
    elif q2_abs >= q0_abs and q2_abs >= q1_abs and q2_abs >= q3_abs:
        q2 = q2_abs
        q0 = (R[0, 2] - R[2, 0]) / (4 * q2)
        q1 = (R[0, 1] + R[1, 0]) / (4 * q2)
        q3 = (R[1, 2] + R[2, 1]) / (4 * q2)
    else:
        q3 = q3_abs
        q0 = (R[1, 0] - R[0, 1]) / (4 * q3)
        q1 = (R[0, 2] + R[2, 0]) / (4 * q3)
        q2 = (R[1, 2] + R[2, 1]) / (4 * q3)

    q = torch.Tensor([q0, q1, q2, q3])
    return q
