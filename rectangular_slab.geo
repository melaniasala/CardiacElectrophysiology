// Define the dimensions of the rectangular slab
h = 3.0;  // height
w = 7.0;  // width
l = 20.0;  // length

// Create the rectangular slab
Point(1) = {0, 0, 0, 1.0};
Point(2) = {w, 0, 0, 1.0};
Point(3) = {w, l, 0, 1.0};
Point(4) = {0, l, 0, 1.0};
Point(5) = {0, 0, h, 1.0};
Point(6) = {w, 0, h, 1.0};
Point(7) = {w, l, h, 1.0};
Point(8) = {0, l, h, 1.0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};
Line(9) = {1, 5};
Line(10) = {2, 6};
Line(11) = {3, 7};
Line(12) = {4, 8};

Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
Line Loop(2) = {5, 6, 7, 8};
Plane Surface(2) = {2};
Line Loop(3) = {1, 10, -5, -9};
Plane Surface(3) = {3};
Line Loop(4) = {2, 11, -6, -10};
Plane Surface(4) = {4};
Line Loop(5) = {3, 12, -7, -11};
Plane Surface(5) = {5};
Line Loop(6) = {4, 9, -8, -12};
Plane Surface(6) = {6};

Surface Loop(1) = {1, 2, 3, 4, 5, 6};
Volume(1) = {1};

// Physical entities
Physical Volume(1) = {1};

