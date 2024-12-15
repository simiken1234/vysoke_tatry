import xml.etree.ElementTree as ET
import re
import matplotlib.pyplot as plt


def add_header(gcode_filename):
    # Add header to GCode file from the file header_ce3pro.gcode
    gcode_file = open(gcode_filename, 'w')
    header_file = open('gcode_parts/header_ce3pro.gcode', 'r')
    for line in header_file:
        gcode_file.write(line)
    header_file.close()
    gcode_file.close()


def add_footer(gcode_filename):
    # Add footer to GCode file from the file footer_ce3pro.gcode
    gcode_file = open(gcode_filename, 'a')
    footer_file = open('gcode_parts/footer_ce3pro.gcode', 'r')
    for line in footer_file:
        gcode_file.write(line)
    footer_file.close()
    gcode_file.close()


def visualize_paths(paths, canvas_extents, home):
    plt.figure()
    for path in paths:
        xs, ys = zip(*path)
        xs = [x - home[0] for x in xs]
        ys = [y - home[1] for y in ys]
        plt.plot(xs, ys)

    plt.xlim(0, canvas_extents[0])
    plt.ylim(0, canvas_extents[1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('SVG to GCode Path Visualization')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.grid(True)
    plt.show()


class Command:
    def __init__(self, x, y, write=False):
        self.x = x
        self.y = y
        self.write = write


class Style:
    def __init__(self, color, width):
        self.color = color  # Hex string
        self.width = width  # Float

    def comp(self, other_color, other_width):
        return self.color == other_color and self.width == other_width


class StyledCommands:
    def __init__(self, style):
        self.style = style
        self.commands = []
        self.x_offset = 0

    def set_x_offset(self, x_offset):
        self.x_offset = x_offset


def convert_svg_to_gcode(svg_filename, gcode_filename, canvas_origin, canvas_extents, margins, calibration_point, z_hop, center=True, keep_aspect=True, visualize=False):
    """
    Convert an SVG file to a GCode file
    :param svg_filename: Name of the SVG file
    :param gcode_filename: Name of the GCode file
    :param extents: Maximum dimensions of the drawing area in mm as [x_extent, y_extent]
    :param keep_aspect: Whether to keep the SVG aspect ratio
    :param visualize: Whether to visualize the output paths
    :return: None
    """
    x_extent, y_extent = canvas_extents[0] - 2 * margins[0], canvas_extents[1] - 2 * margins[1]

    # Add header to GCode file
    add_header(gcode_filename)

    svg_tree = ET.parse(svg_filename)
    root = svg_tree.getroot()

    gcode_file = open(gcode_filename, 'a')

    # Run through commands once just to find the actual size of drawing (not canvas)
    xs, ys, strokes = [], [], []
    for path in root.iter('{http://www.w3.org/2000/svg}path'):
        style = path.get('style')
        if 'fill:none' in style or 'fill: none' in style:
            d = path.get('d')
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

    drawing_origin = [canvas_origin[0] + margins[0], canvas_origin[1] + margins[1]]
    if center:
        x_offset = (x_extent - svg_width * x_scale) / 2
        y_offset = (y_extent - svg_height * y_scale) / 2
        drawing_origin[0] += x_offset
        drawing_origin[1] += y_offset

    # Load all commands

    vis_paths = []
    styled_commands_groups = []
    i_styled_commands = -1
    last_style = None  # Raw style string
    stroke_pattern = r"stroke:\s*([^;]+);"
    stroke_width_pattern = r"stroke-width:\s*([^;]+);"
    for path in root.iter('{http://www.w3.org/2000/svg}path'):
        style = path.get('style')
        if 'fill:none' in style or 'fill: none' in style:
            if style != last_style:
                stroke_match = re.search(stroke_pattern, style)
                stroke_width_match = re.search(stroke_width_pattern, style)
                color = stroke_match.group(1) if stroke_match else None
                width = stroke_width_match.group(1) if stroke_width_match else None
                # Compare to all styles of styled_commands_groups to see if it matches, if not create a new one
                i_found = None
                for i, styled_commands in enumerate(styled_commands_groups):
                    if styled_commands.style.comp(color, width):
                        i_found = i
                        break
                if i_found is not None:
                    i_styled_commands = i_found
                else:
                    i_styled_commands = len(styled_commands_groups)
                    styled_commands_groups.append(StyledCommands(Style(color, width)))
            d = path.get('d')
            vis_path_coords = []
            commands = re.findall('([ML])\s*((?:-?\d+\.?\d*\s+){2})', d)
            for command, coords in commands:
                x, y = coords.strip().split()
                # Apply scaling and translation to position paths within the specified extents
                x = (float(x) - min_x) * x_scale
                y = (max_y - float(y)) * y_scale
                x += drawing_origin[0]
                y += drawing_origin[1]
                vis_path_coords.append((x, y))
                if command == 'M':
                    styled_commands_groups[i_styled_commands].commands.append(Command(x, y, False))
                    #gcode_file.write(f'G91\nG0 Z 1.5\nG90\nG0 X{x} Y{y}\nG91\nG0 Z -1.5\nG90')
                elif command == 'L':
                    styled_commands_groups[i_styled_commands].commands.append(Command(x, y, True))
                    #gcode_file.write(f'G0 X{x} Y{y}\n')
            vis_paths.append(vis_path_coords)

    # Write GCode

    #calibration_point = [canvas_origin[0] + calibration_point[0], canvas_origin[1] + calibration_point[1]]
    gcode_file.write(f'G90\nG0 Z {canvas_origin[2] + 10}\n')
    for i, styled_commands in enumerate(styled_commands_groups):
        # Move to calibration_point, print the style on the board, pause, execute all for the style, repeat.
        # Move to calibration point to put pen in holder
        print(f'{i+1}. color: {styled_commands.style.color}, width: {styled_commands.style.width}')
        gcode_file.write(f'G90\nG0 X{calibration_point[0]} Y{calibration_point[1]}\n')
        gcode_file.write(f'G0 Z {canvas_origin[2]}\n')
        # Write style to display
        gcode_file.write(f'M117 C:{styled_commands.style.color} W:{styled_commands.style.width}\n')
        # Vibrate to signal start of new style
        gcode_file.write(f'M400\nM300 S300 P300\n')
        gcode_file.write(f'M400\nG4 P500\n')
        # Pause print
        gcode_file.write(f'M400\nM0\n')
        # Print resuming printing
        gcode_file.write(f'M117 Printing\n')
        # Move the pen off the paper
        gcode_file.write(f'G91\nG0 Z {canvas_origin[2] + 10}\nG90\n')
        # Move to the first point
        gcode_file.write(f'G0 X{styled_commands.commands[0].x} Y{styled_commands.commands[0].y} Z{canvas_origin[2] + 10}\n')
        gcode_file.write(f'G0 Z{canvas_origin[2]}\n')
        # Cycle through all commands
        for command in styled_commands.commands:
            x_offset = 0
            if styled_commands.style.color == '#808080':
                x_offset = 0
            if command.write:
                gcode_file.write(f'G0 X{command.x + x_offset} Y{command.y}\n')
            else:
                gcode_file.write(f'G91\nG0 Z {z_hop}\nG90\nG0 X{command.x + x_offset} Y{command.y}\nG91\nG0 Z {-z_hop}\nG90\n')
        # Move the pen off the paper
        gcode_file.write(f'G91\nG0 Z {canvas_origin[2] + 10}\nG90\n')
    # Move the pen off the paper
    gcode_file.write(f'G91\nG0 Z {canvas_origin[2] + 10}\nG90\n')

    gcode_file.close()
    add_footer(gcode_filename)

    if visualize:
        visualize_paths(vis_paths, canvas_extents, canvas_origin)


if __name__ == '__main__':
    convert_svg_to_gcode(
        'generated_svg/triglav_3.svg',
        'generated_gcode/triglav_3.gcode',
        canvas_origin=[38, 17, 20],
        canvas_extents=[100, 150],
        margins=[8, 8],
        calibration_point=[90, 235],
        z_hop=2,
        center=True,
        keep_aspect=True,
        visualize=True
    )
