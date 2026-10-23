class _O:
    def __init__(self, xyxy, name, score):
        x1, y1, x2, y2 = xyxy
        self.bbox = type("B", (), {"to_xyxy": lambda s: [x1, y1, x2, y2]})()
        self.category = type("C", (), {"name": name})()
        self.score = type("S", (), {"value": score})()


class _Res:
    def __init__(self, objs):
        self.object_prediction_list = objs


def get_sliced_prediction(path, dm, slice_height=320, slice_width=320, force_postprocess_type=False, **kw):
    boost = {480: 0.15, 320: 0.3, 160: 0.5}.get(slice_height, 0.2)
    r = dm.model.predict(path, conf=0.01, recall_boost=boost)[0]
    b = r.boxes
    return _Res([_O(x, r.names[int(c)], float(s)) for x, s, c in zip(b.xyxy.a, b.conf.a, b.cls.a)])
