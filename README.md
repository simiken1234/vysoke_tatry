# How to use

### 1. Download area of interest from OpenTopography
- Go to [OpenTopography](https://opentopography.org/)
- Select the area of interest
- Select a dataset (30m resolution is plenty good)
- Select the GeoTiff format
- Do not select any of the tick-boxes

### 2. Transform to gpx in QGIS
- Open QGIS
- Add layer
  - Add raster layer
  - Select the GeoTiff file
  - Add
- Create contours
  - In the top panel click 'Raster' -> 'Extraction' -> 'Contour'
  - Pick a spacing for the contour lines - bigger spacing less time to process. 20m should be fine.
  - Click 'Run'
- Export the Layer
  - In the left panel right click on the new layer with the contours and click 'Export' -> 'Save Features As...'
  - Select 'GPX' as the format
  - Select 'GPX_USE_EXTENSIONS' option
  - Select 'include z-dimension' option

### 3. GPX to SVG
- In main.py change the input file to the gpx file you just created
- Change the output file name as well
- Run the script, select a good point of view, then hard-code that elev and azimuth in the script
- Run the script again with the save option enabled

### 4. Set up the printer
- Turn it on, home the axis
- Measure the paper, or your desired dimensions, write it down
- Attach paper to print bed. Even one clip is enough but more is better
- Move z to about 15mm, attach the pen to the print head
- Find the corner of the paper, write down the coordinates
- Adjust pen so that its touching the paper

### 4. SVG to GCODE
- Put in the origin as the home and the dimensions as the extent
- Put in the name of the svg file
- Adjust name for gcode
- Run the script