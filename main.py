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


def add_terrain_plot(ax, xx, yy, z, plot_type='wireframe', **kwargs):
    """Add a terrain plot (contours, wireframe, or surface) to a given axis."""
    if plot_type == 'wireframe':
        ax.plot_wireframe(xx, yy, z, color=kwargs.get('color', 'black'), linewidth=kwargs.get('linewidth', 0.5),
                          rstride=kwargs.get('rstride', 10), cstride=kwargs.get('cstride', 10), antialiased=True)
    elif plot_type == 'surface':
        ax.plot_surface(xx, yy, z, color=kwargs.get('color', 'white'), linewidth=kwargs.get('linewidth', 1),
                        antialiased=kwargs.get('antialiased', False), shade=kwargs.get('shade', False))
    elif plot_type == 'contour':
        contour_spacing = kwargs.get('contour_spacing', 30)  # [m]
        contour_levels = np.arange(np.nanmin(z), np.nanmax(z), contour_spacing)
        ax.contour(xx, yy, z, contour_levels, colors=kwargs.get('colors', 'k'), linewidths=0.5)
        ax.plot_wireframe(xx, yy, z, color=kwargs.get('color', 'black'), linewidth=kwargs.get('linewidth', 0.5),
                          rcount=1, ccount=1, antialiased=True)
    else:
        raise ValueError("Invalid plot_type. Choose 'wireframe', 'surface', or 'contour'.")


def add_path_plot(ax, path_points, **kwargs):
    """Add a GPS path to the terrain plot."""
    ax.plot(path_points[:, 1], path_points[:, 0], path_points[:, 2], color=kwargs.get('color', 'red'), linewidth=1.5, zorder=10)


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


def plot_terrain(gpx_contours_path=None, gpx_path_path=None, resolution=90, plot_type='wireframe', save_path=None, **kwargs):
    """Main function to load data, generate grid, and create a 3D plot."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('xx')
    ax.set_ylabel('yy')
    ax.set_zlabel('z')
    ax.view_init(elev=60, azim=-148)

    contour_points, origin_lat, origin_lon = load_gpx_contours(gpx_contours_path)
    xx, yy, z = generate_grid(contour_points, resolution=resolution)
    xx, yy, z = crop_grid(xx, yy, z, crop_xx=[2000, 7500], crop_yy=[3500, 9000])  # Optionally comment out
    add_terrain_plot(ax, xx, yy, z, plot_type=plot_type, **kwargs)

    if gpx_path_path:
        path_points = load_gpx_path(gpx_path_path, origin_lat, origin_lon, xx, yy, z)
        add_path_plot(ax, path_points, **kwargs)

    set_axis_scaling(ax, xx, yy, z, z_exaggeration=kwargs.get('z_exaggeration', 1.0))

    if save_path:
        ax.set_facecolor('white')
        ax.grid(False)
        ax.set_axis_off()
        plt.savefig(save_path, format='svg')
    else:
        plt.show()


if __name__ == '__main__':
    plot_terrain(
        gpx_contours_path='topography_gpx/triglav_contours.gpx',
        gpx_path_path='path_gpx/triglav_path.gpx',
        resolution=30,
        contour_spacing=75,
        plot_type='contour',
        save_path='generated_svg/triglav.svg'
    )
