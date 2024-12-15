import gpxpy
import gpxpy.gpx
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, RegularGridInterpolator
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import colorchooser
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import os
import pickle
import rasterio


def load_terrain(file_path, resolution=None, **kwargs):
    """
    Load and process geospatial data from GPX or GeoTIFF files, with caching using pickle.

    Args:
        file_path (str): Path to the input file (.gpx or .tif).
        resolution (int, optional): Desired resolution for grid generation.
        **kwargs: Additional arguments for cropping and processing.

    Returns:
        tuple: (xx, yy, z, origin_lat, origin_lon)
    """
    pickle_path = file_path.replace('.gpx', '.pkl').replace('.tif', '.pkl')

    # Check if pickle file exists
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
        print('Loaded from pickle')
        return data['xx'], data['yy'], data['z'], data['origin_lat'], data['origin_lon']

    # Load data based on file type
    if file_path.lower().endswith('.gpx'):
        # Load GPX data
        with open(file_path, 'r') as gpx_file:
            gpx = gpxpy.parse(gpx_file)
        points = []
        for route in gpx.routes:
            elevation = float(route.extensions[2].text)
            for point in route.points:
                points.append([point.latitude, point.longitude, elevation])
        points = np.array(points)

        # Transform latitude and longitude to meters
        origin_lat = np.min(points[:, 0])
        origin_lon = np.min(points[:, 1])
        mean_lat = np.mean(points[:, 0])
        points[:, 0] = (points[:, 0] - origin_lat) * 111000  # 1 degree latitude = 111 km
        points[:, 1] = (points[:, 1] - origin_lon) * 111000 * np.cos(mean_lat * np.pi / 180)

        xx, yy, z = generate_grid(points, resolution=resolution)

    elif file_path.lower().endswith(('.tif', '.tiff')):
        # Load GeoTIFF data
        with rasterio.open(file_path) as src:
            z = src.read(1)  # Read the first band
            transform = src.transform

            # Generate coordinates
            rows, cols = z.shape
            xx, yy = np.meshgrid(
                np.linspace(transform[2], transform[2] + transform[0] * cols, cols),
                np.linspace(transform[5], transform[5] + transform[4] * rows, rows)
            )

            # Convert GeoTIFF coordinates to meters relative to the origin
            origin_lat = np.min(yy)
            origin_lon = np.min(xx)
            mean_lat = np.mean(yy)
            xx = (xx - origin_lon) * 111000 * np.cos(mean_lat * np.pi / 180)
            yy = (yy - origin_lat) * 111000

    else:
        raise ValueError("Unsupported file format. Supported formats are .gpx and .tif/.tiff.")

    # Save to pickle for future use
    data = {'xx': xx, 'yy': yy, 'z': z, 'origin_lat': origin_lat, 'origin_lon': origin_lon}
    with open(pickle_path, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)

    return xx, yy, z, origin_lat, origin_lon


def load_gpx_path(file_path, origin_lat, origin_lon, xx, yy, z):
    """
    Load and process GPX path data into a point array, with caching using pickle.
    """
    pickle_path = file_path.replace('.gpx', '.pkl')

    # Check if pickle file exists
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as pkl_file:
            points = pickle.load(pkl_file)
        return points

    # Load and process GPX file
    with open(file_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)

    points = []
    for point in gpx.tracks[0].segments[0].points:
        points.append([point.latitude, point.longitude, point.elevation])
    points = np.array(points)

    # Transform latitude and longitude to meters relative to the given origin
    mean_lat = np.mean(points[:, 0])
    points[:, 0] = (points[:, 0] - origin_lat) * 111000  # 1 degree latitude = 111 km
    points[:, 1] = (points[:, 1] - origin_lon) * 111000 * np.cos(mean_lat * np.pi / 180)

    # Adjust elevations using the grid data
    interpolator = RegularGridInterpolator((yy[:, 0], xx[0, :]), z, bounds_error=False, fill_value=np.nan)
    points[:, 2] = interpolator((points[:, 0], points[:, 1]))

    # Save to pickle for future use
    with open(pickle_path, 'wb') as pkl_file:
        pickle.dump(points, pkl_file)

    return points


def generate_grid(points, resolution=90):
    """Generate a grid of x, y coordinates and interpolate elevation values."""
    x = np.arange(np.min(points[:, 1]), np.max(points[:, 1]), resolution)
    y = np.arange(np.min(points[:, 0]), np.max(points[:, 0]), resolution)
    xx, yy = np.meshgrid(x, y)
    z = griddata((points[:, 1], points[:, 0]), points[:, 2], (xx, yy), method='cubic')
    return xx, yy, z


def crop_grid(xx, yy, z, crop_xx, crop_yy):
    """Crop a grid to a given bounding box."""
    mask_x = np.logical_and(xx[0, :] >= crop_xx[0], xx[0, :] <= crop_xx[1])
    mask_y = np.logical_and(yy[:, 0] >= crop_yy[0], yy[:, 0] <= crop_yy[1])
    # Apply the mask to crop the grid
    cropped_xx = xx[np.ix_(mask_y, mask_x)]
    cropped_yy = yy[np.ix_(mask_y, mask_x)]
    cropped_z = z[np.ix_(mask_y, mask_x)]
    return cropped_xx, cropped_yy, cropped_z


def add_terrain_plot(ax, xx, yy, z, **kwargs):
    """Add a terrain plot (contours, wireframe, or surface) to a given axis."""
    contour_spacing = kwargs.get('contour_spacing', 30)  # [m]
    contour_levels = np.arange(np.nanmin(z), np.nanmax(z), contour_spacing)
    ax.contour(xx, yy, z, contour_levels,
               colors=kwargs.get('contours_color', 'black'), linewidths=kwargs.get('contours_width', 0.5))
    ax.plot_wireframe(xx, yy, z,
                      color=kwargs.get('borders_color', 'black'), linewidth=kwargs.get('borders_width', 0.5),
                      rcount=1, ccount=1, antialiased=True)


def add_path_plot(ax, path_points, **kwargs):
    """Add a GPS path to the terrain plot."""
    ax.plot(path_points[:, 1], path_points[:, 0], path_points[:, 2],
            color=kwargs.get('path_color', 'black'), linewidth=kwargs.get('path_width', 0.5), zorder=10)


def add_radial_lines(ax, xx, yy, z, num_lines=8, **kwargs):
    """Add radial lines following the elevation of the surface.

    Args:
        ax (Axes3D): Matplotlib 3D axis.
        xx (ndarray): X coordinates of the grid.
        yy (ndarray): Y coordinates of the grid.
        z (ndarray): Elevation values of the grid.
        num_lines (int): Number of radial lines to plot.
    """
    # Find the highest point on the map
    max_idx = np.unravel_index(np.nanargmax(z), z.shape)
    center_x, center_y, center_z = xx[max_idx], yy[max_idx], z[max_idx]

    # Create an interpolator for fast elevation lookups
    interpolator = RegularGridInterpolator((yy[:, 0], xx[0, :]), z, bounds_error=False, fill_value=np.nan)

    # Define angles for radial lines
    angles = np.linspace(0, 2 * np.pi, num_lines, endpoint=False)

    for angle in angles:
        # Compute radial line endpoints in the xy plane
        direction_x = np.cos(angle)
        direction_y = np.sin(angle)

        # Initialize points along the line
        line_x, line_y, line_z = [center_x], [center_y], [center_z]

        # Step outward in the direction of the angle
        while True:
            # Move a small step outward
            new_x = line_x[-1] + direction_x * (xx[0, 1] - xx[0, 0])  # Use grid resolution for step size
            new_y = line_y[-1] + direction_y * (yy[1, 0] - yy[0, 0])

            # Check if new point is within grid bounds
            if not (np.min(xx) <= new_x <= np.max(xx) and np.min(yy) <= new_y <= np.max(yy)):
                break  # Stop if outside the grid

            # Interpolate elevation for the new point
            new_z = interpolator((new_y, new_x))
            if np.isnan(new_z):
                break  # Stop if elevation data is unavailable

            line_x.append(new_x)
            line_y.append(new_y)
            line_z.append(new_z)

        # Plot the radial line
        ax.plot(line_x, line_y, line_z,
                color=kwargs.get('radial_lines_color', 'black'), linewidth=kwargs.get('radial_lines_width', 0.5))


def set_axis_scaling(ax, xx, yy, z, z_exaggeration=1.0):
    """Set axis scaling for a 3D plot with optional z-axis exaggeration."""
    max_range = np.max([
        np.max(xx) - np.min(xx),
        np.max(yy) - np.min(yy),
        (np.nanmax(z) - np.nanmin(z)) * z_exaggeration
    ]) / 2.0

    mid_x = (np.max(xx) + np.min(xx)) * 0.5
    mid_y = (np.max(yy) + np.min(yy)) * 0.5
    mid_z = (np.nanmax(z) + np.nanmin(z)) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range / z_exaggeration, mid_z + max_range / z_exaggeration)


def plot_terrain(ax, gpx_contours_path=None, gpx_path_path=None, resolution=None, save_path=None, **kwargs):
    """Main function to load data, generate grid, and create a 3D plot."""
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('xx')
    ax.set_ylabel('yy')
    ax.set_zlabel('z')
    ax.view_init(elev=kwargs.get('elev'), azim=kwargs.get('azim'))

    print("Loading contour data...")
    xx, yy, z, origin_lat, origin_lon = load_terrain(gpx_contours_path, resolution, **kwargs)
    if 'crop_xx' in kwargs and 'crop_yy' in kwargs:
        xx, yy, z = crop_grid(xx, yy, z, crop_xx=kwargs.get('crop_xx'), crop_yy=kwargs.get('crop_yy'))
    print("Plotting terrain...")
    add_terrain_plot(ax, xx, yy, z, **kwargs)
    print("Plotting radial lines...")
    add_radial_lines(ax, xx, yy, z, num_lines=kwargs.get('n_radials', 10), **kwargs)

    if gpx_path_path:
        print("Loading path data...")
        path_points = load_gpx_path(gpx_path_path, origin_lat, origin_lon, xx, yy, z)
        print("Plotting path...")
        add_path_plot(ax, path_points, **kwargs)

    print("Setting axis scaling...")
    set_axis_scaling(ax, xx, yy, z, z_exaggeration=kwargs.get('z_exaggeration', 1.0))

    ax.set_facecolor(kwargs.get('background_color', 'white'))
    if not kwargs.get('show_axes', True):
        ax.grid(False)
        ax.set_axis_off()

    print("Showing/saving plot...")
    if save_path:
        plt.savefig(save_path, format='svg')
    else:
        plt.show()


def create_interactive_window():
    """Create a Tkinter window for interactive color adjustment."""
    def update_plot():
        """Update the plot with new colors."""
        ax.clear()  # Clear the current plot
        plot_terrain(
            ax=ax,
            # Paths
            gpx_contours_path='topography_gpx/gerlach.tif',
            gpx_path_path='path_gpx/gerlach.gpx',
            # save_path='generated_svg/triglav_2.svg',  # Comment out if saving not desired
            # Terrain viz settings
            resolution=30,  # [m]
            contour_spacing=float(contour_spacing.get()),  # [m]
            n_radials=int(num_radials.get()),
            # Slavkovsky crop
            # crop_xx=[2000, 7000],  # [m]
            # crop_yy=[1000, 6000],  # [m]
            # Gerlach crop
            crop_xx=[1500, 5000],  # [m]
            crop_yy=[2500, 6000],  # [m]
            # Triglav crop
            # crop_xx=[2000, 7500],  # [m]
            # crop_yy=[3500, 9000],  # [m]
            z_exaggeration=1.0,  # Vertical exaggeration, 1.0 = no exaggeration
            # Colors
            path_color=path_color.get(),
            radial_lines_color=radial_lines_color.get(),
            contours_color=contours_color.get(),
            borders_color=borders_color.get(),
            background_color=background_color.get(),
            # Line widths
            path_width=float(path_width.get()),
            radial_lines_width=float(radial_lines_width.get()),
            contours_width=float(contours_width.get()),
            borders_width=float(borders_width.get()),
            # Plot settings
            show_axes=False,
            elev=38,
            azim=75
        )
        canvas.draw()  # Redraw the updated plot

    def save_plot():
        """Save the current plot as an SVG."""
        file_path = 'generated_svg/gerlach.svg'  # Define the output file name
        fig.savefig(file_path, format='svg')
        print(f"Plot saved as {file_path}")

    def pick_color(variable):
        """Open a color chooser and set the selected color to the variable."""
        color = colorchooser.askcolor()[1]  # Get the selected color
        if color:
            variable.set(color)  # Update the Tkinter variable
            update_plot()  # Update the plot with the new color

    def pick_color_from_image(variable):
        """Open an image, allow the user to pick a color by clicking on it, and update the plot."""
        def on_click(event):
            # Get the color of the clicked pixel
            x, y = int(event.x), int(event.y)
            rgb = image.getpixel((x, y))
            hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            variable.set(hex_color)  # Update the color variable
            update_plot()  # Update the plot with the new color
            img_window.destroy()  # Close the image window after picking

        file_path = 'palette.jpg'

        # Open and display the image
        img_window = tk.Toplevel(root)
        img_window.title("Pick a Color from Image")
        image = Image.open(file_path)
        tk_image = ImageTk.PhotoImage(image)
        img_label = tk.Label(img_window, image=tk_image)
        img_label.image = tk_image
        img_label.pack()

        # Bind click event to get color
        img_label.bind("<Button-1>", on_click)

    def update_view_label(event):
        """Update the view angle label with the current elevation and azimuth."""
        elev = ax.elev
        azim = ax.azim
        view_label.config(text=f"Elevation: {elev:.1f}°, Azimuth: {azim:.1f}°")

    # Create Tkinter window
    root = tk.Tk()
    root.title("Interactive Terrain Plot")

    # Create Matplotlib figure and axis
    fig = plt.Figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Embed Matplotlib figure in Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Add a label to display the current view angles
    view_label = tk.Label(root, text="Elevation: 0°, Azimuth: 0°", anchor='e')
    view_label.pack(side=tk.BOTTOM, fill=tk.X)

    # Connect Matplotlib's draw event to update the label
    fig.canvas.mpl_connect("draw_event", update_view_label)

    # Define Tkinter variables for colors and parameters
    path_color = tk.StringVar(value='gray')
    radial_lines_color = tk.StringVar(value='red')
    contours_color = tk.StringVar(value='lightgray')
    borders_color = tk.StringVar(value='black')
    background_color = tk.StringVar(value='white')
    num_radials = tk.StringVar(value='15')
    contour_spacing = tk.StringVar(value='100.0')
    path_width = tk.StringVar(value='0.5')
    radial_lines_width = tk.StringVar(value='0.5')
    contours_width = tk.StringVar(value='0.5')
    borders_width = tk.StringVar(value='0.5')

    # Controls frame for organizing options
    controls_frame = tk.Frame(root)
    controls_frame.pack(side=tk.BOTTOM, fill=tk.X)

    # First row: Radial lines and contour spacing
    row1 = tk.Frame(controls_frame)
    row1.pack(side=tk.TOP, fill=tk.X)

    tk.Label(row1, text="Radial Lines:").pack(side=tk.LEFT)
    tk.Entry(row1, textvariable=num_radials, width=5).pack(side=tk.LEFT)

    tk.Label(row1, text="Contour Spacing:").pack(side=tk.LEFT)
    tk.Entry(row1, textvariable=contour_spacing, width=5).pack(side=tk.LEFT)

    # Second row: Line widths
    row2 = tk.Frame(controls_frame)
    row2.pack(side=tk.TOP, fill=tk.X)

    tk.Label(row2, text="Radial Line Width:").pack(side=tk.LEFT)
    tk.Entry(row2, textvariable=radial_lines_width, width=5).pack(side=tk.LEFT)

    tk.Label(row2, text="Path Line Width:").pack(side=tk.LEFT)
    tk.Entry(row2, textvariable=path_width, width=5).pack(side=tk.LEFT)

    tk.Label(row2, text="Border Line Width:").pack(side=tk.LEFT)
    tk.Entry(row2, textvariable=borders_width, width=5).pack(side=tk.LEFT)

    tk.Label(row2, text="Contour Line Width:").pack(side=tk.LEFT)
    tk.Entry(row2, textvariable=contours_width, width=5).pack(side=tk.LEFT)

    # Third row: Color pickers
    row3 = tk.Frame(controls_frame)
    row3.pack(side=tk.TOP, fill=tk.X)

    tk.Label(row3, text="Path Color:").pack(side=tk.LEFT)
    tk.Button(row3, text="Pick from Image", command=lambda: pick_color_from_image(path_color)).pack(side=tk.LEFT)

    tk.Label(row3, text="Radial Lines Color:").pack(side=tk.LEFT)
    tk.Button(row3, text="Pick from Image", command=lambda: pick_color_from_image(radial_lines_color)).pack(side=tk.LEFT)

    tk.Label(row3, text="Contours Color:").pack(side=tk.LEFT)
    tk.Button(row3, text="Pick from Image", command=lambda: pick_color_from_image(contours_color)).pack(side=tk.LEFT)

    tk.Label(row3, text="Borders Color:").pack(side=tk.LEFT)
    tk.Button(row3, text="Pick from Image", command=lambda: pick_color_from_image(borders_color)).pack(side=tk.LEFT)

    tk.Label(row3, text="Background Color:").pack(side=tk.LEFT)
    tk.Button(row3, text="Pick from Image", command=lambda: pick_color_from_image(background_color)).pack(side=tk.LEFT)

    # Fourth row: Redraw button, Save button
    row4 = tk.Frame(controls_frame)
    row4.pack(side=tk.TOP, fill=tk.X)

    tk.Button(row4, text="Redraw", command=update_plot).pack(side=tk.LEFT)
    tk.Button(row4, text="Save as SVG", command=save_plot).pack(side=tk.LEFT)

    # Initial plot
    update_plot()

    # Run the Tkinter event loop
    root.mainloop()


if __name__ == '__main__':
    create_interactive_window()
