import math
def angle_difference(angle1, angle2):
    """计算两个角度之间的差距。 input angle范围 -180*180"""
    # 将角度转换到0°至360°范围内
    normalized_angle1 = angle1 % 360
    normalized_angle2 = angle2 % 360

    # 计算两个角度之间的差距
    diff = abs(normalized_angle1 - normalized_angle2)

    # 如果差距大于180°，则减去360°得到最短差距
    if diff > 180:
        diff = 360 - diff

    return diff

def calculate_rotation(angle1, angle2, target_difference=100):
    """计算并返回旋转角度a需要旋转的度数，以使得a和b的差值为target_difference"""
    normalized_angle1 = angle1 % 360
    normalized_angle2 = angle2 % 360

    # 计算两个角度之间的差距
    diff = abs(normalized_angle1 - normalized_angle2)

    # 如果差距大于180°，则减去360°得到最短差距
    if diff > 180:
        diff = 360 - diff
    if diff <= target_difference:
        return angle1
    else:
        if angle_difference(normalized_angle2 + target_difference, normalized_angle1) <  angle_difference(normalized_angle2 - target_difference, normalized_angle1):
            angel = normalized_angle2 + target_difference
        else:
            angel = normalized_angle2 - target_difference
    angel = (angel + 180) % 360 - 180
    return angel


# 示例
a = 1
b = 180
print(calculate_rotation(a, b))  # 输出旋转的角度