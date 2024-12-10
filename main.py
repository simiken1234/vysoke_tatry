import gpxpy
import gpxpy.gpx
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, RegularGridInterpolator
from mpl_toolkits.mplot3d import Axes3D


def load_gpx_contours(file_path):
    """Load and process GPX data into a point array."""
    gpx_file = open(file_path, 'r')
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
    return points, origin_lat, origin_lon


def load_gpx_path(file_path, origin_lat, origin_lon, xx, yy, z):
    """Load and process GPX path data into a point array, using the same origin as the contours and adjusting elevations."""
    gpx_file = open(file_path, 'r')
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


def plot_terrain(gpx_contours_path=None, gpx_path_path=None, resolution=90, save_path=None, **kwargs):
    """Main function to load data, generate grid, and create a 3D plot."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('xx')
    ax.set_ylabel('yy')
    ax.set_zlabel('z')
    ax.view_init(elev=kwargs.get('elev'), azim=kwargs.get('azim'))

    print("Loading contour data...")
    contour_points, origin_lat, origin_lon = load_gpx_contours(gpx_contours_path)
    xx, yy, z = generate_grid(contour_points, resolution=resolution)
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

    if not kwargs.get('show_axes', True):
        ax.set_facecolor('white')
        ax.grid(False)
        ax.set_axis_off()

    print("Showing/saving plot...")
    if save_path:
        plt.savefig(save_path, format='svg')
    else:
        plt.show()


if __name__ == '__main__':
    plot_terrain(
        # Paths
        gpx_contours_path='topography_gpx/triglav_contours.gpx',
        gpx_path_path='path_gpx/triglav_path.gpx',
        #save_path='generated_svg/triglav_2.svg',  # Comment out if saving not desired
        # Terrain viz settings
        resolution=30,  # [m]
        contour_spacing=100,  # [m]
        n_radials=15,
        crop_xx=[2000, 7500],  # [m]
        crop_yy=[3500, 9000],  # [m]
        z_exaggeration=1.0,  # Vertical exaggeration, 1.0 = no exaggeration
        # Colors
        path_color='yellow',
        radial_lines_color='magenta',
        contours_color='deepskyblue',
        borders_color='black',
        # Line widths
        path_width=0.5,
        radial_lines_width=0.5,
        contours_width=0.5,
        borders_width=0.5,
        # Plot settings
        show_axes=False,
        elev=38,
        azim=75
    )
