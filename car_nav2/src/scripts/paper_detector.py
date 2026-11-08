#!/usr/bin/env python3
"""Detect the standing A4 sheet (model `mypaper`) in a 2D laser scan.

The scan is split into segments wherever consecutive points jump apart. A segment is
reported as the paper when it matches the sheet's footprint in the scan plane:

  * width   - distance between its end points is close to the A4 width (0.210 m);
              the end points are the sheet's edges, so this holds at any viewing angle
              (except edge-on, where the sheet is invisible to the lidar anyway)
  * flat    - the points lie on a straight line (rejects round things like the fire
              hydrant and the corners of boxes)
  * in front - both neighbouring segments are clearly farther away, i.e. the sheet
              stands free in front of the background (a sheet flat against a wall
              cannot be told apart from the wall)

Every received scan is also republished on `output_topic`, so nodes that expect /scan
(SLAM, AMCL, Nav2, RViz) keep working while the bridge publishes /subbascan. In the
republished scan the beams that hit the paper are removed, so those nodes do not see
the paper as an obstacle. By default each removed beam is given the distance of the
background behind the paper: a straight line between the scan points just left and
right of the paper. Those beams then clear an earlier mark of the paper from costmaps
and the SLAM map. With fill_background false they are set to +inf instead
("no return" per REP 117), which only clears in costmaps with inf_is_valid true.

A second copy with the paper always removed (whatever exclude_paper is) is published on
`slam_topic` for slam_toolbox. The SLAM map never records the paper, so switching
exclude_paper only changes what Nav2 sees: slam_toolbox keeps every scan it has
added, so a paper that got into the map would stay in it (and in the global
costmap's static layer) for the rest of the run.

Parameters (ros2 run car_nav2 paper_detector --ros-args -p name:=value):
  scan_topic        topic to read                         (default: subbascan)
  output_topic      topic to republish every scan on;
                    empty string disables republishing    (default: scan)
  exclude_paper     remove the paper's points from the
                    republished scan                      (default: true)
  slam_topic        topic for the scan that never shows the
                    paper (for SLAM); empty disables      (default: scan_slam)
  fill_background   give removed beams the background
                    distance instead of +inf              (default: true)
  paper_width       expected width [m]                    (default: 0.210)
  width_tolerance   accepted +/- error on the width [m]   (default: 0.05)
  max_line_rms      max RMS distance from a line [m]      (default: 0.02)
  min_points        min scan points on the sheet          (default: 4)
  jump_distance     gap that splits two segments [m]      (default: 0.10)
  print_period      seconds between repeated prints while
                    the paper stays in view               (default: 1.0)
"""

import copy
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


def fit_line_rms(xs, ys):
    """RMS perpendicular distance of points from their best-fit (total least squares) line."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    # Smallest eigenvalue of the covariance matrix = residual variance across the line.
    tr, det = sxx + syy, sxx * syy - sxy * sxy
    lam_min = tr / 2.0 - math.sqrt(max(tr * tr / 4.0 - det, 0.0))
    return math.sqrt(max(lam_min, 0.0) / n)


def find_paper(scan, paper_width, width_tol, max_rms, min_points, jump):
    """Return a list of detections: dicts with range, bearing, width, rms, points."""
    pts = []  # (index, x, y, r) for valid returns only
    for i, r in enumerate(scan.ranges):
        if math.isfinite(r) and scan.range_min <= r <= scan.range_max:
            a = scan.angle_min + i * scan.angle_increment
            pts.append((i, r * math.cos(a), r * math.sin(a), r))
    if len(pts) < min_points:
        return []

    # Split into segments at range jumps or gaps of missing beams.
    segments, cur = [], [pts[0]]
    for p, q in zip(pts, pts[1:]):
        if q[0] - p[0] > 2 or math.hypot(q[1] - p[1], q[2] - p[2]) > jump:
            segments.append(cur)
            cur = []
        cur.append(q)
    segments.append(cur)

    # A 360 deg scan wraps around: merge the last segment into the first if they touch.
    full_circle = abs(scan.angle_increment * len(scan.ranges)) >= 2 * math.pi - 2 * abs(scan.angle_increment)
    if full_circle and len(segments) > 1:
        a, b = segments[-1][-1], segments[0][0]
        if b[0] + len(scan.ranges) - a[0] <= 2 and math.hypot(a[1] - b[1], a[2] - b[2]) <= jump:
            segments[0] = segments.pop() + segments[0]

    detections = []
    n = len(segments)
    for k, seg in enumerate(segments):
        if len(seg) < min_points or n < 3:
            continue
        width = math.hypot(seg[-1][1] - seg[0][1], seg[-1][2] - seg[0][2])
        if abs(width - paper_width) > width_tol:
            continue
        rms = fit_line_rms([p[1] for p in seg], [p[2] for p in seg])
        if rms > max_rms:
            continue
        # Must stand in front of what is on both sides of it.
        near = min(p[3] for p in seg)
        prev_r, next_r = segments[k - 1][-1][3], segments[(k + 1) % n][0][3]
        if not (prev_r > near + jump and next_r > near + jump):
            continue
        cx = sum(p[1] for p in seg) / len(seg)
        cy = sum(p[2] for p in seg) / len(seg)
        detections.append({
            'range': math.hypot(cx, cy),
            'bearing': math.atan2(cy, cx),
            'x': cx, 'y': cy,
            'width': width, 'rms': rms, 'points': len(seg),
            'indices': [p[0] for p in seg],
        })
    return detections


class PaperDetector(Node):

    def __init__(self):
        super().__init__('paper_detector')
        self.declare_parameter('scan_topic', 'subbascan')
        self.declare_parameter('paper_width', 0.210)
        self.declare_parameter('width_tolerance', 0.05)
        self.declare_parameter('max_line_rms', 0.02)
        self.declare_parameter('min_points', 4)
        self.declare_parameter('jump_distance', 0.10)
        self.declare_parameter('print_period', 1.0)

        self.declare_parameter('output_topic', 'scan')
        self.declare_parameter('exclude_paper', True)
        self.declare_parameter('fill_background', True)
        self.declare_parameter('slam_topic', 'scan_slam')

        topic = self.get_parameter('scan_topic').value
        out_topic = self.get_parameter('output_topic').value
        slam_topic = self.get_parameter('slam_topic').value
        for name, t in (('output_topic', out_topic), ('slam_topic', slam_topic)):
            if t == topic:
                raise ValueError(f'{name} must differ from scan_topic ("{topic}") '
                                 'or every scan would loop back into this node')
        if out_topic and out_topic == slam_topic:
            raise ValueError(f'output_topic and slam_topic must differ ("{out_topic}")')
        # Reliable QoS: works for both reliable and best-effort subscribers of /scan.
        self.pub = self.create_publisher(LaserScan, out_topic, 10) if out_topic else None
        self.slam_pub = self.create_publisher(LaserScan, slam_topic, 10) if slam_topic else None
        if self.slam_pub:
            self.get_logger().info(f'Publishing "{topic}" on "{slam_topic}" with the paper '
                                   'always removed (for SLAM).')
        self.sub = self.create_subscription(LaserScan, topic, self.on_scan, qos_profile_sensor_data)
        if self.pub:
            excl = ''
            if self.get_parameter('exclude_paper').value:
                fill = ('the background distance' if self.get_parameter('fill_background').value
                        else '+inf')
                excl = f' with the paper replaced by {fill}'
            self.get_logger().info(f'Republishing "{topic}" on "{out_topic}"{excl}.')
        self.in_view = False
        self.last_print = None
        self.get_logger().info(f'Watching "{topic}" for the A4 paper (mypaper)...')

    @staticmethod
    def background_ranges(scan, ranges, indices, search=3):
        """Background distance for each beam in `indices` (one contiguous paper segment).

        The background is the straight line between the nearest valid returns just
        outside the segment on each side (looking up to `search` beams further out).
        Each beam gets the distance at which it would cross that line. If only one side
        has a return, its range is used for all beams; if neither has one, +inf.
        """
        n = len(ranges)
        full_circle = abs(scan.angle_increment * n) >= 2 * math.pi - 2 * abs(scan.angle_increment)

        def valid(i):
            r = ranges[i]
            return math.isfinite(r) and scan.range_min <= r <= scan.range_max

        def neighbour(start, step):
            for k in range(1, search + 2):
                i = start + step * k
                if full_circle:
                    i %= n
                elif not 0 <= i < n:
                    return None
                if i in idx_set:
                    return None
                if valid(i):
                    return i
            return None

        idx_set = set(indices)
        angle = lambda i: scan.angle_min + i * scan.angle_increment  # noqa: E731
        left, right = neighbour(indices[0], -1), neighbour(indices[-1], +1)
        if left is None and right is None:
            return [math.inf] * len(indices)
        if left is None or right is None:
            r = ranges[left if right is None else right]
            return [r] * len(indices)

        # Line through the two background points; intersect each beam's ray with it.
        ax, ay = ranges[left] * math.cos(angle(left)), ranges[left] * math.sin(angle(left))
        bx, by = ranges[right] * math.cos(angle(right)), ranges[right] * math.sin(angle(right))
        dx, dy = bx - ax, by - ay
        out = []
        for j, i in enumerate(indices):
            ux, uy = math.cos(angle(i)), math.sin(angle(i))
            denom = ux * dy - uy * dx
            r = (ax * dy - ay * dx) / denom if abs(denom) > 1e-9 else -1.0
            if not r > 0.0:
                # Ray parallel to / pointing away from the line: interpolate by beam index.
                t = (j + 1) / (len(indices) + 1)
                r = ranges[left] + t * (ranges[right] - ranges[left])
            out.append(min(max(r, scan.range_min), scan.range_max))
        return out

    @classmethod
    def without_paper(cls, scan, dets, fill_background=True):
        """Return the scan with every beam that hit a detected paper replaced by the
        background distance behind it (or +inf when fill_background is false)."""
        if not dets:
            return scan
        original = list(scan.ranges)
        ranges = list(original)
        intensities = list(scan.intensities)
        for d in dets:
            idx = d['indices']
            fill = (cls.background_ranges(scan, original, idx) if fill_background
                    else [math.inf] * len(idx))
            for i, r in zip(idx, fill):
                ranges[i] = r
                if i < len(intensities):
                    intensities[i] = 0.0
        scan.ranges = ranges
        scan.intensities = intensities
        return scan

    def on_scan(self, scan):
        g = lambda name: self.get_parameter(name).value  # noqa: E731
        dets = find_paper(scan, g('paper_width'), g('width_tolerance'), g('max_line_rms'),
                          g('min_points'), g('jump_distance'))
        # Paper-free copy; without_paper replaces the range lists, so copy the message
        # first to keep `scan` itself unchanged.
        filtered = (self.without_paper(copy.deepcopy(scan), dets, g('fill_background'))
                    if dets else scan)
        if self.pub:
            self.pub.publish(filtered if g('exclude_paper') else scan)
        if self.slam_pub:
            self.slam_pub.publish(filtered)
        now = self.get_clock().now()

        if not dets:
            if self.in_view:
                self.get_logger().info('Paper lost from view.')
            self.in_view = False
            return

        due = (self.last_print is None or not self.in_view or
               (now - self.last_print).nanoseconds * 1e-9 >= g('print_period'))
        self.in_view = True
        if not due:
            return
        self.last_print = now
        for d in dets:
            self.get_logger().info(
                f'PAPER DETECTED in {scan.header.frame_id}: range {d["range"]:.2f} m, '
                f'bearing {math.degrees(d["bearing"]):+.1f} deg '
                f'(x={d["x"]:.2f}, y={d["y"]:.2f}), width {d["width"] * 100:.1f} cm, '
                f'{d["points"]} pts, line rms {d["rms"] * 1000:.1f} mm')


def main(args=None):
    rclpy.init(args=args)
    node = PaperDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
