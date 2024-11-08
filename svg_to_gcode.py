import xml.etree.ElementTree as ET
import re
import matplotlib.pyplot as plt


def add_header(gcode_filename):
    # Add header to GCode file from the file header_ce3pro.gcode
    gcode_file = open(gcode_filename, 'w')
    header_file = open('header_ce3pro.gcode', 'r')
    for line in header_file:
        gcode_file.write(line)
    header_file.close()
    gcode_file.close()


def add_footer(gcode_filename):
    # Add footer to GCode file from the file footer_ce3pro.gcode
    gcode_file = open(gcode_filename, 'a')
    footer_file = open('footer_ce3pro.gcode', 'r')
    for line in footer_file:
        gcode_file.write(line)
    footer_file.close()
    gcode_file.close()


def visualize_paths(paths, area):
    plt.figure()
    x_min_desired, y_min_desired = area[0]
    x_max_desired, y_max_desired = area[1]

    for path in paths:
        xs, ys = zip(*path)
        plt.plot(xs, ys)

    #plt.xlim(x_min_desired, x_max_desired)
    #plt.ylim(y_min_desired, y_max_desired)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('SVG to GCode Path Visualization')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.grid(True)
    plt.show()


def convert_svg_to_gcode(svg_filename, gcode_filename, area, keep_aspect=True, visualize=False):
    """
    Convert an SVG file to a GCode file
    :param svg_filename: Name of the SVG file
    :param gcode_filename: Name of the GCode file
    :param area: Area of the drawing in mm. Format: [[x_min_desired, y_min_desired], [x_max_desired, y_max_desired]]
    :param keep_aspect: Do you want to keep the aspect ratio of the SVG file?, bool
    :return: None
    """
    assert area[0][0] < area[1][0] and area[0][1] < area[1][1], 'Area must be a rectangle'
    assert area[0][0] >= 0 and area[0][1] >= 0, 'Area must be positive'

    # Add header to GCode file
    add_header(gcode_filename)

    svg_tree = ET.parse(svg_filename)
    root = svg_tree.getroot()

    gcode_file = open(gcode_filename, 'a')

    x_printer = 25
    y_printer = 0#25

    x_pen = 15
    y_pen = 45
    z_pen = 35.1

    gcode_file.write(f'G0 Z{z_pen+5}\n')
    strokes = []
    xs = []
    ys = []

    for path in root.iter('{http://www.w3.org/2000/svg}path'):
        style = path.get('style')
        if 'fill:none' in style or 'fill: none' in style:
            d = path.get('d')
            if 'stroke-width' in style:
                stroke = float(style.split('stroke-width:')[1][1:])
                strokes.append(stroke)

            commands = re.findall('([ML])\s*((?:-?\d+\.?\d*\s+){2})', d)
            for command, coords in commands:
                x, y = coords.strip().split()
                xs.append(float(x))
                ys.append(float(y))

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    width = max_x - min_x
    height = max_y - min_y

    #print(f'Min x: {min_x}, Max x: {max_x}, Min y: {min_y}, Max y: {max_y}')

    max_stroke = max(strokes)
    min_stroke = min(strokes)
    z_min_stroke = z_pen
    z_max_stroke = z_pen

    z_hop = 1.5

    # Get scaling and translation factors
    x_min_desired, y_min_desired = area[0]
    x_max_desired, y_max_desired = area[1]
    x_scale = (x_max_desired - x_min_desired) / width
    y_scale = (y_max_desired - y_min_desired) / height
    if keep_aspect:
        scale = min(x_scale, y_scale)
        x_scale = y_scale = scale
    x_offset = x_min_desired - (min_x * x_scale) + x_pen + x_printer
    y_offset = y_min_desired - (min_y * y_scale) + y_pen + y_printer

    paths = []
    for path in root.iter('{http://www.w3.org/2000/svg}path'):
        style = path.get('style')
        if 'fill:none' in style or 'fill: none' in style:
            d = path.get('d')
            if 'stroke-width' in style and max_stroke != min_stroke:
                stroke = float(style.split('stroke-width:')[1][1:])
                z_height = z_min_stroke + (stroke - min_stroke) * (z_max_stroke - z_min_stroke) / \
                           (max_stroke - min_stroke)
            else:
                z_height = z_max_stroke
            gcode_file.write(f'G0 Z{z_height}\n')

            path_coords = []
            commands = re.findall('([ML])\s*((?:-?\d+\.?\d*\s+){2})', d)
            for command, coords in commands:
                x, y = coords.strip().split()
                x = (float(x) * x_scale + x_offset)
                y = (float(y) * y_scale + y_offset)
                path_coords.append((x, y))
                if command == 'M':
                    gcode_file.write(f'G91\nG0 Z {z_hop}\nG90\nG0 X{x} Y{y}\nG91\nG0 Z {-z_hop}\nG90')
                elif command == 'L':
                    gcode_file.write(f'G1 X{x} Y{y}\n')

            paths.append(path_coords)

    gcode_file.close()

    add_footer(gcode_filename)

    if visualize:
        visualize_paths(paths, area)


if __name__ == '__main__':
    convert_svg_to_gcode('tatry_2.svg', 'tatry_2_test.gcode', area=[[30, 30], [160, 210]], keep_aspect=True, visualize=True)
