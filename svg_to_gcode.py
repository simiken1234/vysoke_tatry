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


def visualize_paths(paths, x_extent, y_extent):
    plt.figure()
    for path in paths:
        xs, ys = zip(*path)
        plt.plot(xs, ys)

    # plt.xlim(0, x_extent)
    # plt.ylim(0, y_extent)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('SVG to GCode Path Visualization')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.grid(True)
    plt.show()


def convert_svg_to_gcode(svg_filename, gcode_filename, home, extents, keep_aspect=True, visualize=False):
    """
    Convert an SVG file to a GCode file
    :param svg_filename: Name of the SVG file
    :param gcode_filename: Name of the GCode file
    :param extents: Maximum dimensions of the drawing area in mm as [x_extent, y_extent]
    :param keep_aspect: Whether to keep the SVG aspect ratio
    :param visualize: Whether to visualize the output paths
    :return: None
    """
    x_extent, y_extent = extents
    assert x_extent > 0 and y_extent > 0, 'Extents must be positive values'

    # Add header to GCode file
    add_header(gcode_filename)

    svg_tree = ET.parse(svg_filename)
    root = svg_tree.getroot()

    gcode_file = open(gcode_filename, 'a')

    gcode_file.write(f'\nG0 Z{home[2]}\n')

    xs, ys, strokes = [], [], []

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

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    svg_width, svg_height = max_x - min_x, max_y - min_y

    x_scale = x_extent / svg_width
    y_scale = y_extent / svg_height
    if keep_aspect:
        scale = min(x_scale, y_scale)
        x_scale = y_scale = scale
    print(x_scale, y_scale)

    paths = []
    for path in root.iter('{http://www.w3.org/2000/svg}path'):
        style = path.get('style')
        if 'fill:none' in style or 'fill: none' in style:
            d = path.get('d')

            path_coords = []
            commands = re.findall('([ML])\s*((?:-?\d+\.?\d*\s+){2})', d)
            for command, coords in commands:
                x, y = coords.strip().split()
                # Apply scaling and translation to position paths within the specified extents
                x = (float(x) - min_x) * x_scale
                y = (max_y - float(y)) * y_scale
                x += home[0]
                y += home[1]
                path_coords.append((x, y))
                if command == 'M':
                    gcode_file.write(f'G91\nG0 Z 1.5\nG90\nG0 X{x} Y{y}\nG91\nG0 Z -1.5\nG90')
                elif command == 'L':
                    gcode_file.write(f'G0 X{x} Y{y}\n')

            paths.append(path_coords)

    gcode_file.close()
    add_footer(gcode_filename)

    if visualize:
        visualize_paths(paths, x_extent, y_extent)


if __name__ == '__main__':
    convert_svg_to_gcode('damascus_30.svg', 'damascus_30_2.gcode', home=[70, 32, 20], extents=[119, 75], keep_aspect=True, visualize=True)  #area=[[30, 30], [160, 210]]
