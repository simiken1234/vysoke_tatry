M82 ;absolute extrusion mode
; Ender 3 Custom Start G-code
G28 ; Home all axes
G1 Z2.0 F3000 ; Move Z Axis up little to prevent scratching of Heat Bed
M107 P0 ; Turn off fan
M107 P1 ; Turn off fan
M84 X Y E ;Disable all steppers but Z
G90