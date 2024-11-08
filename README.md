# cardioEMI
Solving cell-by-cell (a.k.a. EMI) models on cardiac geometries.

### Dependencies

* FEniCSx (www.fenicsproject.org)
* multiphenicsx (https://github.com/multiphenics/multiphenicsx)


### Download

```
git clone https://github.com/pietrobe/cardioEMI.git
cd cardioEMI
```

### Installation via Docker

To install FEniCSx:

```
docker run -t -v $(pwd):/home/fenics -i ghcr.io/fenics/dolfinx/dolfinx:v0.7.1
cd /home/fenics
```

To install multiphenicsx in the docker container:

```
pip install multiphenicsx@git+https://github.com/multiphenics/multiphenicsx.git@dolfinx-v0.7.1
```

### Testing the installation

```
mpirun -n 1 python3 -u main.py input.yml
```
modifying *input.yml* fordifferent input data. Parallel execution can be obtain with mpirun -n X.

### Mesh creation 
An square input mesh can be created via 

```
python3 create_square_mesh.py
```
in *create_square_mesh.py* geometric settings (#elements and #cells) can be modified.

###  Visualize output in Paraview
Given an output file *solX.xdmf* and an tags mesh file *tags.xdmf*, apply the following filters:
+ `Extract blocks` to *tags.xdmf* of cell_tag data, followed by `Merge blocks`
+ `Append Attributes` of both *solX.xdmf* and the MergeBlock (order can be important)
+ `Threshold` to the appended attribute according to cell tag and field of interest

Similarly can be done for the membrane output file *v.xdmf* and facet tags.

