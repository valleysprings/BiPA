"""Layout check of a drawn figure: every visible text at least 6.5 pt, inside the canvas, and no two text boxes overlapping."""
import matplotlib


def check(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
    bad, boxes = [], []
    for t in texts:
        if t.get_fontsize() < 6.5:
            bad.append(("small", t.get_text(), t.get_fontsize()))
        bb = t.get_window_extent(renderer)
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > fig.bbox.width + 0.5 or bb.y1 > fig.bbox.height + 0.5:
            bad.append(("clipped", t.get_text()))
        boxes.append((t.get_text(), bb))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                bad.append(("overlap", boxes[i][0], boxes[j][0]))
    for b in bad:
        print("layout problem:", b)
    return bad
