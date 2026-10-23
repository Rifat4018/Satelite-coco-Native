def get_slice_bboxes(image_height, image_width, slice_height, slice_width, overlap_height_ratio=0.2, overlap_width_ratio=0.2):
    out = []
    yo, xo = int(overlap_height_ratio * slice_height), int(overlap_width_ratio * slice_width)
    y = 0
    while y < image_height:
        x = 0
        y2 = y + slice_height
        while x < image_width:
            x2 = x + slice_width
            out.append([x, y, x2, y2])
            if x2 >= image_width:
                break
            x = x2 - xo
        if y2 >= image_height:
            break
        y = y2 - yo
    return out
