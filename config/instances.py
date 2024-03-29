yolov8_instance = None
arcface_inswapper_instance = None
face_occluder_instance = None
face_parser_instance = None
codeformer_instance = None
inswapper_instance = None


def clear_instances():
    global yolov8_instance
    global arcface_inswapper_instance
    global face_occluder_instance
    global face_parser_instance
    global codeformer_instance
    global inswapper_instance
    yolov8_instance = None
    arcface_inswapper_instance = None
    face_occluder_instance = None
    face_parser_instance = None
    codeformer_instance = None
    inswapper_instance = None