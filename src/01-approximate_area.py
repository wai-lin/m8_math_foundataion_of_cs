import math


def is_inside(x, y):
    return math.sqrt(x**2 + y**2) <= 1


def approx_area(p1, p2, p3, p4):
    a, b = (p1, p2)
    c, d = (p3, p4)
    n = 1500

    h = (b-a)/n
    x_segs = []
    y_segs = []
    for i in range(n+1):
        x_segs.append(a + i*h)
        y_segs.append(c + i*h)

    inside_count = 0
    on_edge_count = 0
    for i in range(n):
        for j in range(n):
            x1, y1 = x_segs[i], y_segs[j]
            x2, y2 = x_segs[i], y_segs[j+1]
            x3, y3 = x_segs[i+1], y_segs[j]
            x4, y4 = x_segs[i+1], y_segs[j+1]

            if is_inside(x1, y1) and is_inside(x2, y2) and is_inside(x3, y3) and is_inside(x4, y4):
                inside_count += 1
            elif is_inside(x1, y1) or is_inside(x2, y2) or is_inside(x3, y3) or is_inside(x4, y4):
                on_edge_count += 1

    rect_area = h * h
    inside_area = inside_count * rect_area
    on_edge_area = on_edge_count * rect_area
    approx_area = inside_area + (on_edge_area / 2)

    print(f"For points, ({p1}, {p2}), ({p3}, {p4})")
    print("Inside count:", inside_count)
    print("On edge count:", on_edge_count)
    print(f"Approximate area: {approx_area:.2f}")
    print("==================================================\n")


approx_area(-1, 1, -1, 1)
approx_area(-1, 3, -1, 3)
approx_area(-2, 4, -2, 4)
