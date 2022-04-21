from utilities import get_intersection_area

class GateParams:
    """
    :type intersection_method: int
    :type intersection_thresh: float
    """

    def __init__(self):
        self.intersection_method = 2
        self.intersection_thresh = 0.1
        self.help = {
            'intersection_method': 'method used for computing gate-target intersection: '
                                   '0: box center lies on gate '
                                   '1: gate intersects box'
                                   '2: gate intersects target trajectory',
            'intersection_thresh': 'threshold for box center intersection with gate',
        }


class Gate:
    def __init__(self, params, _id, pt1, pt2, logger):
        """
        :type params: GateParams
        :type _id: int
        :type pt1: tuple(int, int)
        :type pt2: tuple(int, int)
        :type logger: logging.RootLogger
        :rtype: None
        """

        self.params = params
        self._id = _id
        self.pt1 = pt1
        self.pt2 = pt2
        self.logger = logger

        x0, y0 = self.pt1
        x1, y1 = self.pt2

        if x0 < x1:
            self.xmin = x0
            self.xmax = x1
        else:
            self.xmax = x0
            self.xmin = x1

        if y0 < y1:
            self.ymin = y0
            self.ymax = y1
        else:
            self.ymax = y0
            self.ymin = y1

        self.range = self.xmax - self.xmin
        self.yrange = self.ymax - self.ymin

        self.xdiff = x1 - x0
        self.ydiff = y1 - y0

        self.is_vert = self.xdiff == 0
        self.is_horz = self.ydiff == 0

        self.x0, self.y0 = x0, y0
        self.x1, self.y1 = x1, y1

        self.box = [self.xmin, self.ymin, self.xmax, self.ymax]

        self.intersections = {}

    def isSame(self, pt1, pt2):
        return pt1[0] == self.pt1[0] and pt1[1] == self.pt1[1] and \
               pt2[0] == self.pt2[0] and pt2[1] == self.pt2[1]

    # check if two points lie on different sides of the gate
    def onDifferentSides(self, pt1, pt2):
        if self.is_vert:
            return (pt1[0] < self.x0) == (pt2[0] > self.x0)
        elif self.is_horz:
            return (pt1[1] < self.y0) == (pt2[1] > self.y0)
        else:
            val_x0 = float(pt1[0] - self.x0) / self.xdiff
            val_y0 = float(pt1[1] - self.y0) / self.ydiff

            val_x1 = float(pt2[0] - self.x0) / self.xdiff
            val_y1 = float(pt2[1] - self.y0) / self.ydiff

            return (val_x0 < val_y0) == (val_x1 > val_y1)

    def updateIntersections(self, bboxes, frame_id):

        for _id in bboxes:
            if _id in self.intersections:
                # each target can only intersect with a gate once
                continue

            box = bboxes[_id]

            if self.params.intersection_method == 2:
                pt1, pt2 = box
                if not self.onDifferentSides(pt1, pt2):
                    continue

                x0, y0 = pt1
                x1, y1 = pt2
                xdiff = x1 - x0
                ydiff = y1 - y0

                if xdiff == 0:
                    intersects = ((self.pt1[0] < x0) == (self.pt2[0] > x0))
                elif ydiff == 0:
                    intersects = ((self.pt1[1] < y0) == (self.pt2[1] > y0))
                else:
                    val_x0 = float(self.pt1[0] - x0) / xdiff
                    val_y0 = float(self.pt1[1] - y0) / ydiff

                    val_x1 = float(self.pt2[0] - x0) / xdiff
                    val_y1 = float(self.pt2[1] - y0) / ydiff

                    intersects = ((val_x0 < val_y0) == (val_x1 > val_y1))
            else:
                box_xmin, box_ymin = box[0]
                box_xmax, box_ymax = box[1]

                box_2 = [box_xmin, box_ymin, box_xmax, box_ymax]

                if get_intersection_area(self.box, box_2) <= 0:
                    continue

                if self.params.intersection_method == 0:
                    box_xmid = float(box_xmax + box_xmin) / 2.0
                    box_ymid = float(box_ymax + box_ymin) / 2.0
                    if self.is_vert:
                        intersects = (abs(box_xmid - self.xmin) <= self.params.intersection_thresh )
                    elif self.is_horz:
                        intersects = (abs(box_ymid - self.ymin) <= self.params.intersection_thresh)
                    else:
                        val_xmid = float(box_xmid - self.x0) / self.xdiff
                        val_ymid = float(box_ymid - self.y0) / self.ydiff

                        intersects = (abs(val_xmid - val_ymid) <= self.params.intersection_thresh)
                else:

                    if self.is_vert:
                        intersects = ((box_xmin - self.xmin) * (box_xmax - self.xmin) < 0)
                    elif self.is_horz:
                        intersects = ((box_ymin - self.ymin) * (box_ymax - self.ymin) < 0)
                    else:
                        val_xmin = float(box_xmin - self.x0) / self.xdiff
                        val_ymin = float(box_ymin - self.y0) / self.ydiff

                        val_xmax = float(box_xmax - self.x0) / self.xdiff
                        val_ymax = float(box_ymax - self.y0) / self.ydiff

                        ul_val = val_xmin - val_ymin
                        ur_val = val_xmax - val_ymin
                        br_val = val_xmax - val_ymax
                        bl_val = val_xmin - val_ymax

                        # either of the two pairs of diagonally opposite points of the bb
                        # are on different sides of the gate
                        intersects = ((ul_val * br_val < 0) or (ur_val * bl_val < 0))

            if intersects:
                self.logger.info('Target {:d} passed through gate {:d} in frame {:d}'.format(
                    _id, self._id, frame_id))
                # print('self.pts: ', self.pts)
                # print('box: ', box)
                self.intersections[_id] = frame_id