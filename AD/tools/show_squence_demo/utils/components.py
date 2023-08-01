import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw

from colorsys import rgb_to_yiq

class LabelLUT:
    """The class to manage look-up table for assigning colors to labels."""

    class Label:
        def __init__(self, name, value, color):
            self.name = name
            self.value = value
            self.color = color

    Colors = [[0., 0., 0.], [0.96078431, 0.58823529, 0.39215686],
              [0.96078431, 0.90196078, 0.39215686],
              [0.58823529, 0.23529412, 0.11764706],
              [0.70588235, 0.11764706, 0.31372549], [1., 0., 0.],
              [0.11764706, 0.11764706, 1.], [0.78431373, 0.15686275, 1.],
              [0.35294118, 0.11764706, 0.58823529], [1., 0., 1.],
              [1., 0.58823529, 1.], [0.29411765, 0., 0.29411765],
              [0.29411765, 0., 0.68627451], [0., 0.78431373, 1.],
              [0.19607843, 0.47058824, 1.], [0., 0.68627451, 0.],
              [0., 0.23529412,
               0.52941176], [0.31372549, 0.94117647, 0.58823529],
              [0.58823529, 0.94117647, 1.], [0., 0., 1.], [1.0, 1.0, 0.25],
              [0.5, 1.0, 0.25], [0.25, 1.0, 0.25], [0.25, 1.0, 0.5],
              [0.25, 1.0, 1.25], [0.25, 0.5, 1.25], [0.25, 0.25, 1.0],
              [0.125, 0.125, 0.125], [0.25, 0.25, 0.25], [0.375, 0.375, 0.375],
              [0.5, 0.5, 0.5], [0.625, 0.625, 0.625], [0.75, 0.75, 0.75],
              [0.875, 0.875, 0.875]]

    def __init__(self, label_to_names=None):
        """
        Args:
            label_to_names: Initialize the colormap with this mapping from
                labels (int) to class names (str).
        """
        self._next_color = 10
        self.labels = {}
        if label_to_names is not None:
            for val in sorted(label_to_names.keys()):
                self.add_label(label_to_names[val], val)

    def add_label(self, name, value, color=None):
        """Adds a label to the table.

        Example:
            The following sample creates a LUT with 3 labels::

                lut = ml3d.vis.LabelLUT()
                lut.add_label('one', 1)
                lut.add_label('two', 2)
                lut.add_label('three', 3, [0,0,1]) # use blue for label 'three'

        Args:
            name: The label name as string.
            value: The value associated with the label.
            color: Optional RGB color. E.g., [0.2, 0.4, 1.0].
        """
        if color is None:
            if self._next_color >= len(self.Colors):
                self._next_color = 0
                color = self.Colors[self._next_color]
                self._next_color += 1
            else:
                color = self.Colors[self._next_color]
                self._next_color += 1
        self.labels[value] = self.Label(name, value, color)

    @classmethod
    def get_colors(self, name='default', mode=None):
        """Return full list of colors in the lookup table.

        Args:
            name (str): Name of lookup table colormap. Only 'default' is
                supported.
            mode (str): Colormap mode. May be None (return as is), 'lightbg" to
                move the dark colors earlier in the list or 'darkbg' to move
                them later in the list. This will provide better visual
                discrimination for the earlier classes.

        Returns:
            List of colors (R, G, B) in the LUT.
        """
        if mode is None:
            return self.Colors
        dark_colors = list(
            filter(lambda col: rgb_to_yiq(*col)[0] < 0.5, self.Colors))
        light_colors = list(
            filter(lambda col: rgb_to_yiq(*col)[0] >= 0.5, self.Colors))
        if mode == 'lightbg':
            return dark_colors + light_colors
        if mode == 'darkbg':
            return light_colors + dark_colors


class BoundingBox3D:
    """Class that defines an axially-oriented bounding box."""

    next_id = 1

    def __init__(self,
                 center,
                 front,
                 up,
                 left,
                 size,
                 label_class,
                 confidence,
                 meta=None,
                 show_class=False,
                 show_confidence=False,
                 show_meta=None,
                 meta_center=None,
                 identifier=None,
                 arrow_length=1.0):
        """Creates a bounding box.

        Front, up, left define the axis of the box and must be normalized and
        mutually orthogonal.

        Args:
            center: (x, y, z) that defines the center of the box.
            front: normalized (i, j, k) that defines the front direction of the
                box.
            up: normalized (i, j, k) that defines the up direction of the box.
            left: normalized (i, j, k) that defines the left direction of the
                box.
            size: (width, height, depth) that defines the size of the box, as
                measured from edge to edge.
            label_class: integer specifying the classification label. If an LUT
                is specified in create_lines() this will be used to determine
                the color of the box.
            confidence: confidence level of the box.
            meta: a user-defined string (optional).
            show_class: displays the class label in text near the box
                (optional).
            show_confidence: displays the confidence value in text near the box
                (optional).
            show_meta: displays the meta string in text near the box (optional).
            identifier: a unique integer that defines the id for the box
                (optional, will be generated if not provided).
            arrow_length: the length of the arrow in the front_direct. Set to
                zero to disable the arrow (optional).
        """
        assert (len(center) == 3)
        assert (len(front) == 3)
        assert (len(up) == 3)
        assert (len(left) == 3)
        assert (len(size) == 3)
        assert (len(meta_center) == 3)

        self.center = np.array(center, dtype="float32")
        self.front = np.array(front, dtype="float32")
        self.up = np.array(up, dtype="float32")
        self.left = np.array(left, dtype="float32")
        self.size = size
        self.label_class = label_class
        self.confidence = confidence
        self.meta = meta
        self.show_class = show_class
        self.show_confidence = show_confidence
        self.show_meta = show_meta
        self.meta_center = meta_center
        if identifier is not None:
            self.identifier = identifier
        else:
            self.identifier = "box:" + str(BoundingBox3D.next_id)
            BoundingBox3D.next_id += 1
        self.arrow_length = arrow_length

    def __repr__(self):
        s = str(self.identifier) + " (class=" + str(
            self.label_class) + ", conf=" + str(self.confidence)
        if self.meta is not None:
            s = s + ", meta=" + str(self.meta)
        s = s + ")"
        return s

    @staticmethod
    def create_lines(boxes, lut=None, out_format="lineset"):
        """Creates a LineSet that can be used to render the boxes.

        Args:
            boxes: the list of bounding boxes
            lut: a ml3d.vis.LabelLUT that is used to look up the color based on
                the label_class argument of the BoundingBox3D constructor. If
                not provided, a color of 50% grey will be used. (optional)
            out_format (str): Output format. Can be "lineset" (default) for the
                Open3D lineset or "dict" for a dictionary of lineset properties.

        Returns:
            For out_format == "lineset": open3d.geometry.LineSet
            For out_format == "dict": Dictionary of lineset properties
                ("vertex_positions", "line_indices", "line_colors", "bbox_labels",
                "bbox_confidences").
        """
        if out_format not in ('lineset', 'dict'):
            raise ValueError("Please specify an output_format of 'lineset' "
                             "(default) or 'dict'.")

        nverts = 14
        nlines = 17
        points = np.zeros((nverts * len(boxes), 3), dtype="float32")
        indices = np.zeros((nlines * len(boxes), 2), dtype="int32")
        colors = np.zeros((nlines * len(boxes), 3), dtype="float32")

        for i, box in enumerate(boxes):
            pidx = nverts * i
            x = 0.5 * box.size[0] * box.left
            y = 0.5 * box.size[1] * box.up
            z = 0.5 * box.size[2] * box.front
            arrow_tip = box.center + z - box.arrow_length * box.front
            # arrow_mid = box.center + z + 0.60 * box.arrow_length * box.front
            # head_length = 0.3 * box.arrow_length
            # It seems to be substantially faster to assign directly for the
            # points, as opposed to points[pidx:pidx+nverts] = np.stack((...))
            points[pidx] = box.center + x + y + z
            points[pidx + 1] = box.center - x + y + z
            points[pidx + 2] = box.center - x + y - z
            points[pidx + 3] = box.center + x + y - z
            points[pidx + 4] = box.center + x - y + z
            points[pidx + 5] = box.center - x - y + z
            points[pidx + 6] = box.center - x - y - z
            points[pidx + 7] = box.center + x - y - z
            points[pidx + 8] = box.center + z
            points[pidx + 9] = arrow_tip
            # points[pidx + 10] = arrow_mid + head_length * box.up
            # points[pidx + 11] = arrow_mid - head_length * box.up
            # points[pidx + 12] = arrow_mid + head_length * box.left
            # points[pidx + 13] = arrow_mid - head_length * box.left
            points[pidx + 10] = arrow_tip
            points[pidx + 11] = arrow_tip
            points[pidx + 12] = arrow_tip
            points[pidx + 13] = arrow_tip

        # It is faster to break the indices and colors into their own loop.
        for i, box in enumerate(boxes):
            pidx = nverts * i
            idx = nlines * i
            indices[idx:idx +
                    nlines] = ((pidx, pidx + 1), (pidx + 1, pidx + 2),
                               (pidx + 2, pidx + 3), (pidx + 3, pidx),
                               (pidx + 4, pidx + 5), (pidx + 5, pidx + 6),
                               (pidx + 6, pidx + 7), (pidx + 7, pidx + 4),
                               (pidx + 0, pidx + 4), (pidx + 1, pidx + 5),
                               (pidx + 2, pidx + 6), (pidx + 3, pidx + 7),
                               (pidx + 8, pidx + 9), (pidx + 9, pidx + 10),
                               (pidx + 9, pidx + 11), (pidx + 9,
                                                       pidx + 12), (pidx + 9,
                                                                    pidx + 13))

            if lut is not None and box.label_class in lut.labels:
                label = lut.labels[box.label_class]
                c = (label.color[0], label.color[1], label.color[2])
            else:
                if box.confidence == -1.0:
                    c = (0., 1.0, 0.)  # GT: Green
                elif box.confidence >= 0 and box.confidence <= 1.0:
                    c = (1.0, 0., 0.)  # Prediction: red
                else:
                    c = (0.5, 0.5, 0.5)  # Grey

            colors[idx:idx +
                   nlines] = c  # copies c to each element in the range
        if out_format == "lineset":
            lines = o3d.geometry.LineSet()
            lines.points = o3d.utility.Vector3dVector(points)
            lines.lines = o3d.utility.Vector2iVector(indices)
            lines.colors = o3d.utility.Vector3dVector(colors)
        elif out_format == "dict":
            lines = {
                "vertex_positions": points,
                "line_indices": indices,
                "line_colors": colors,
                "bbox_labels": tuple(b.label_class for b in boxes),
                "bbox_confidences": tuple(b.confidence for b in boxes)
            }

        return lines

    @staticmethod
    def project_to_img(boxes, img, lidar2img_rt=np.ones(4), lut=None):
        """Returns image with projected 3D bboxes

        Args:
            boxes: the list of bounding boxes
            img: an RGB image
            lidar2img_rt: 4x4 transformation from lidar frame to image plane
            lut: a ml3d.vis.LabelLUT that is used to look up the color based on
                the label_class argument of the BoundingBox3D constructor. If
                not provided, a color of 50% grey will be used. (optional)
        """
        lines = BoundingBox3D.create_lines(boxes, lut, out_format="dict")
        points = lines["vertex_positions"]
        indices = lines["line_indices"]
        colors = lines["line_colors"]

        pts_4d = np.concatenate(
            [points.reshape(-1, 3),
             np.ones((len(boxes) * 14, 1))], axis=-1)
        pts_2d = pts_4d @ lidar2img_rt.T

        pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        imgfov_pts_2d = pts_2d[..., :2].reshape(len(boxes), 14, 2)
        indices_2d = indices[..., :2].reshape(len(boxes), 17, 2)
        colors_2d = colors[..., :3].reshape(len(boxes), 17, 3)

        return BoundingBox3D.plot_rect3d_on_img(img,
                                                len(boxes),
                                                imgfov_pts_2d,
                                                indices_2d,
                                                colors_2d,
                                                thickness=3)

    @staticmethod
    def plot_rect3d_on_img(img,
                           num_rects,
                           rect_corners,
                           line_indices,
                           color=None,
                           thickness=1):
        """Plot the boundary lines of 3D rectangular on 2D images.

        Args:
            img (numpy.array): The numpy array of image.
            num_rects (int): Number of 3D rectangulars.
            rect_corners (numpy.array): Coordinates of the corners of 3D
                rectangulars. Should be in the shape of [num_rect, 8, 2] or
                [num_rect, 14, 2] if counting arrows.
            line_indices (numpy.array): indicates connectivity of lines between
                rect_corners.  Should be in the shape of [num_rect, 12, 2] or
                [num_rect, 17, 2] if counting arrows.
            color (tuple[int]): The color to draw bboxes. Default: (1.0, 1.0,
                1.0), i.e. white.
            thickness (int, optional): The thickness of bboxes. Default: 1.
        """
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)

        if color is None:
            color = np.ones((line_indices.shape[0], line_indices.shape[1], 3))
        for i in range(num_rects):
            corners = rect_corners[i].astype(np.int)
            # ignore boxes outside a certain threshold
            interesting_corners_scale = 3.0
            if min(corners[:, 0]
                  ) < -interesting_corners_scale * img.shape[1] or max(
                      corners[:, 0]
                  ) > interesting_corners_scale * img.shape[1] or min(
                      corners[:, 1]
                  ) < -interesting_corners_scale * img.shape[0] or max(
                      corners[:, 1]) > interesting_corners_scale * img.shape[0]:
                continue
            for j, (start, end) in enumerate(line_indices[i]):
                c = tuple(color[i][j] * 255)  # TODO: not working
                c = (int(c[0]), int(c[1]), int(c[2]))
                if i != 0:
                    pt1 = (corners[(start) % (14 * i),
                                   0], corners[(start) % (14 * i), 1])
                    pt2 = (corners[(end) % (14 * i),
                                   0], corners[(end) % (14 * i), 1])
                else:
                    pt1 = (corners[start, 0], corners[start, 1])
                    pt2 = (corners[end, 0], corners[end, 1])
                draw.line([pt1, pt2], fill=c, width=thickness)
        return np.array(img_pil).astype(np.uint8)


class Object3D(BoundingBox3D):
    def __init__(self,
                 center,
                 size,
                 yaw,
                 name,
                 cls="",
                 arrow=0.,
                 score=0.,
                 id="",
                 text="",
                 thikness=1.5,
                 show_meta=False,
                 meta_center=None,
                 show_arrow=False):

        self.yaw = yaw-np.pi*0.5
        left = [np.cos(self.yaw), np.sin(self.yaw), 0]
        front = [-np.sin(self.yaw), np.cos(self.yaw), 0]
        up = [0, 0, 1]

        self.score = score
        self.cls = cls
        self.name = name
        self.id = id
        self.thikness=thikness
        show_name = self.name

        if show_arrow is False:
            self.arrow = 0.
        else:
            if arrow < 1.0:
                self.arrow = size[2]*0.33
            else: self.arrow = arrow
        
        super().__init__(center, front, up, left, size,
                         label_class=show_name,
                         confidence=self.score,
                         meta=text,
                         show_class=False,
                         show_confidence=False,
                         show_meta=show_meta,
                         meta_center=meta_center,
                         identifier=None,
                         arrow_length=self.arrow)


class Colormap:
    """This class is used to create a color map for visualization of points."""

    class Point:
        """Initialize the class.

        Args:
            value: The scalar value index of the point.
            color: The color associated with the value.
        """

        def __init__(self, value, color):
            assert (value >= 0.0)
            assert (value <= 1.0)

            self.value = value
            self.color = color

        def __repr__(self):
            """Represent the color and value in the colormap."""
            return "Colormap.Point(" + str(self.value) + ", " + str(
                self.color) + ")"

    # The value of each Point must be greater than the previous
    # (e.g. [0.0, 0.1, 0.4, 1.0], not [0.0, 0.4, 0.1, 1.0]
    def __init__(self, points):
        self.points = points

    def calc_u_array(self, values, range_min, range_max):
        """Generate the basic array based on the minimum and maximum range passed."""
        range_width = (range_max - range_min)
        return [
            min(1.0, max(0.0, (v - range_min) / range_width)) for v in values
        ]

    # (This is done by the shader now)
    def calc_color_array(self, values, range_min, range_max):
        """Generate the color array based on the minimum and maximum range passed.

        Args:
            values: The index of values.
            range_min: The minimum value in the range.
            range_max: The maximum value in the range.

        Returns:
            An array of color index based on the range passed.
        """
        u_array = self.calc_u_array(values, range_min, range_max)

        tex = [[1.0, 0.0, 1.0]] * 128
        n = float(len(tex) - 1)
        idx = 0
        for tex_idx in range(0, len(tex)):
            x = float(tex_idx) / n
            while idx < len(self.points) and x > self.points[idx].value:
                idx += 1

            if idx == 0:
                tex[tex_idx] = self.points[0].color
            elif idx == len(self.points):
                tex[tex_idx] = self.points[-1].color
            else:
                p0 = self.points[idx - 1]
                p1 = self.points[idx]
                dist = p1.value - p0.value
                # Calc weights between 0 and 1
                w0 = 1.0 - (x - p0.value) / dist
                w1 = (x - p0.value) / dist
                c = [
                    w0 * p0.color[0] + w1 * p1.color[0],
                    w0 * p0.color[1] + w1 * p1.color[1],
                    w0 * p0.color[2] + w1 * p1.color[2]
                ]
                tex[tex_idx] = c

        return [tex[int(u * n)] for u in u_array]

    # These are factory methods rather than class objects because
    # the user may modify the colormaps that are used.
    @staticmethod
    def make_greyscale():
        """Generate a greyscale colormap."""
        return Colormap([
            Colormap.Point(0.0, [0.0, 0.0, 0.0]),
            Colormap.Point(1.0, [1.0, 1.0, 1.0])
        ])

    @staticmethod
    def make_rainbow():
        """Generate the rainbow color array."""
        return Colormap([
            Colormap.Point(0.000, [0.0, 0.0, 1.0]),
            Colormap.Point(0.125, [0.0, 0.5, 1.0]),
            Colormap.Point(0.250, [0.0, 1.0, 1.0]),
            Colormap.Point(0.375, [0.0, 1.0, 0.5]),
            Colormap.Point(0.500, [0.0, 1.0, 0.0]),
            Colormap.Point(0.625, [0.5, 1.0, 0.0]),
            Colormap.Point(0.750, [1.0, 1.0, 0.0]),
            Colormap.Point(0.875, [1.0, 0.5, 0.0]),
            Colormap.Point(1.000, [1.0, 0.0, 0.0])
        ])
