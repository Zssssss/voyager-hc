def move_control(obj_type=0, id=0, velocity=None, height=None, course=None):
    return {
        "obj_type": obj_type,
        "id": id,
        "type": 1,
        'vel': velocity,
        "height": height,
        "course": course
    }