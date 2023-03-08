import gpxpy
import gpxpy.gpx
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata


def trim_corners(xx, yy, z, trim=300):
    xx = xx[trim:-trim, trim:-trim]
    yy = yy[trim:-trim, trim:-trim]
    z = z[trim:-trim, trim:-trim]
    return xx, yy, z


# Load the GPX file
gpx_file = open('vysoke_tatry.gpx', 'r')
gpx = gpxpy.parse(gpx_file)
points = np.array([[point.latitude, point.longitude, point.elevation] for point in gpx.tracks[0].segments[0].points])

# Create a 2D grid of points from the latitude and longitude values
x = np.linspace(np.min(points[:, 1]), np.max(points[:, 1]), 1000)
y = np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), 1000)
xx, yy = np.meshgrid(x, y)

# Interpolate the elevation values onto the 2D grid
z = griddata((points[:,1], points[:,0]), points[:,2], (xx, yy), method='cubic')

z_solid = np.subtract(z, 10)

# Trim the corners of the grid
xx, yy, z = trim_corners(xx, yy, z)
print(z.shape)

# Plot the contour lines
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_axis_off()

#ax.plot_wireframe(xx, yy, z, color='black', linewidth=0.5, rstride=100, cstride=100, antialiased=False)

'''ax.plot_surface(xx, yy, z_solid, color='white', linewidth=1, antialiased=False, shade=False)'''

# set the contour lines to black and set line width based on altitude value
contour_lines = ax.contour(xx, yy, z, 20, colors='k')
for i in range(len(contour_lines.collections)):
    pass
    #contour_lines.collections[i].set_linewidth((20 - i) / 10)

#print(contour_lines)
#print(contour_lines.collections[0].get_paths()[0].vertices)

ax.view_init(elev=80, azim=69)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Elevation')

# Scale the z-axis
ax.set_zbound(-2000, max(z.flatten()))

plt.show()

#plt.savefig('contours.svg', format='svg')
