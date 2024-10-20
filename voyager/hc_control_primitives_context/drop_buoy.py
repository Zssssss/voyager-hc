def drop_buoy(buoy_type, channel, lat, lon, height=-175, id=0):
    return {
        "id": id,
        'lat':lat,
        "lon": lon,
        "type": 2,
        "buoy_type": buoy_type,
        "height": height,
        "channel": channel
    }
