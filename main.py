import gpxpy
import gpxpy.gpx
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import random


def trim_corners(xx, yy, z, trim=300):
    xx = xx[trim:-trim, trim:-trim]
    yy = yy[trim:-trim, trim:-trim]
    z = z[trim:-trim, trim:-trim]
    return xx, yy, z


# Load the GPX file
gpx_file = open('damascus_dense.gpx', 'r')
gpx = gpxpy.parse(gpx_file)
#points = np.array([[point.latitude, point.longitude, point.elevation] for point in gpx.routes[0].points])  # in gpx.tracks[0].segments[0].points
points = []
for route in gpx.routes:
    elevation = float(route.extensions[2].text)
    for point in route.points:
        points.append([point.latitude, point.longitude, elevation])
points = np.array(points)
print(len(points))

# Transform latitude, longitude to meters, with origin at minimum lat/long values
mean_lat = np.mean(points[:, 0])
points[:, 0] = (points[:, 0] - np.min(points[:, 0])) * 111000  # 1 degree latitude = 111 km
points[:, 1] = (points[:, 1] - np.min(points[:, 1])) * 111000 * np.cos(mean_lat * np.pi / 180)  # 1 degree longitude = 111 km * cos(latitude)

# Define the desired resolution (distance between points)
resolution = 90  # Example resolution; adjust as needed

# Generate x and y grid based on the resolution
x = np.arange(np.min(points[:, 1]), np.max(points[:, 1]), resolution)
y = np.arange(np.min(points[:, 0]), np.max(points[:, 0]), resolution)

# Create the mesh grid
xx, yy = np.meshgrid(x, y)

# Interpolate the elevation values onto the 2D grid
z = griddata((points[:,1], points[:,0]), points[:,2], (xx, yy), method='cubic')

# Reverse the xx axis
#xx = xx[:, ::-1]

z_solid = np.subtract(z, 10)

# Trim the corners of the grid
#xx, yy, z = trim_corners(xx, yy, z, trim=50)
print(z.shape)

# Plot the contour lines
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_axis_off()

#ax.plot_wireframe(xx, yy, z, color='black', linewidth=0.5, rstride=100, cstride=100, antialiased=False)

'''ax.plot_surface(xx, yy, z_solid, color='white', linewidth=1, antialiased=False, shade=False)'''

# set the contour lines to black and set line width based on altitude value
contour_lines = ax.contour(xx, yy, z, 30, colors='k')
# for i in range(len(contour_lines.collections)):
#     pass
    #contour_lines.collections[i].set_linewidth((20 - i) / 10)

ax.view_init(elev=90, azim=-90)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Elevation')

# Define a scaling factor for the z-axis exaggeration
z_exaggeration = 2.5  # Adjust this value to exaggerate (e.g., 1.0 = no exaggeration)

# Scaling
max_range = np.max([np.max(x) - np.min(x), np.max(y) - np.min(y), (np.nanmax(z) - np.nanmin(z)) * z_exaggeration]) / 2.0

mid_x = (np.max(x) + np.min(x)) * 0.5
mid_y = (np.max(y) + np.min(y)) * 0.5
mid_z = (np.nanmax(z) + np.nanmin(z)) * 0.5

# Set limits with the z-axis exaggerated
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range / z_exaggeration, mid_z + max_range / z_exaggeration)

plt.savefig('damascus_30.svg', format='svg')

plt.show()
