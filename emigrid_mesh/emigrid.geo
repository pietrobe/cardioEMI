SetFactory("OpenCASCADE");

DefineConstant[
  nx = {2, Name "Number of cells in X", Min 1, Max 20, Step 1}
  ny = {2, Name "Number of cells in Y", Min 1, Max 20, Step 1}
  nz = {2, Name "Number of cells in Z", Min 1, Max 20, Step 1}
  cl = {100.0, Name "Cell length, units μm", Min 1.0, Max 300.0, Step 5.0 }
  cw = { 40.0, Name "Cell  width, units μm", Min 1.0, Max 100.0, Step 5.0 }
  gl = {  5.0, Name "Intercalated disk length, units μm", Min 1.0, Max 20.0, Step 1.0 }
  gw = { 10.0, Name "Intercalated disk  width, units μm", Min 1.0, Max 30.0, Step 1.0 }
  minlc = {2.0, Name "Min Characteristic length", Min 0.001, Max 100.0, Step 0.01}
  maxlc = {5.0, Name "Max Characteristic length", Min 0.001, Max 100.0, Step 0.01}
];

// myocytes oriented along x direction
Mesh.CharacteristicLengthMin = minlc;
Mesh.CharacteristicLengthMax = maxlc;

Box(1) = {-cl/2.0+gl, -cw/2.0+gl, -cw/2.0+gl, cl-2*gl, cw-2*gl, cw-2*gl};
Box(2) = {-cl/2.0, -gw/2.0, -gw/2.0, cl, gw, gw};  // x-dir
Box(3) = {-gw/2.0, -cw/2.0, -gw/2.0, gw, cw, gw};  // y-dir
Box(4) = {-gw/2.0, -gw/2.0, -cw/2.0, gw, gw, cw};  // z-dir
Box(5) = {-cl/2.0, -cw/2.0, -cw/2.0, cl, cw, cw};  // extra

BooleanUnion(6) = { Volume{1}; Delete;}{ Volume{2,3,4}; Delete;};
BooleanDifference(7) = { Volume{5}; Delete; }{ Volume{6}; };

base[] = {6,7};

all[] = {};
For i In {0:(nx-1)}
  For j In {0:(ny-1)}
    For k In {0:(nz-1)}
      // Compute offset
      tx = i * cl;
      ty = j * cw;
      tz = k * cw;

      If ((i > 0) || (j > 0) || (k > 0))
        // Copy and translate the base cell
        newVol[] = Translate{tx, ty, tz}{ Duplicata{Volume{base[]};} };
        all[] += newVol[];
        //Printf("Created %g %g %g volume", i, j, k);
      EndIf
    EndFor
  EndFor
EndFor

BooleanFragments{ Volume{base[]}; Delete; }{ Volume{all[]}; Delete; }

Physical Volume(1) = {base[0]};
Physical Volume(2) = {base[1]};
For i In {0:#all[]-1}
  Physical Volume(i+3) = {all[i]};
EndFor

