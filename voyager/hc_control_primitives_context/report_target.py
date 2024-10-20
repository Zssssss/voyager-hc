def report_target(id, target_lat, target_lon, target_height, target_course, target_vel,  report_time, obj_type=0, target_id=0):
    return {
        "obj_type": obj_type,
        "id": id,
        "type": 11,
        'target_lat': target_lat,
        "target_lon": target_lon,
        "target_height": target_height,
        "target_course": target_course,
        "target_vel": target_vel,
        "target_id": target_id,
        "report_time": report_time
    }