# cardioEMI
Solving cell-by-cell (a.k.a. EMI) models for cardiac geometries.

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
modifying `input.yml` for different input data. Parallel execution can be obtain with `mpirun -n X`.

### Geometry and tagging
In the input .yml file two input files have to be provided:
- path to an XDMF mesh with volume and facets tags 
- path to a dictionary file containing the connectivity map between cells and facets

Each volume tag correspond to a FEM space, thus it makes sense to choose the minimum number of volume tags, so that there are no neighbour cells with the same tag. The ECS_TAG can be provided in the input .yml file, otherwise the minimum between all the volume tags will be used. 

The *geometry* directory contains scripts to generate tagged meshes and connectivity dictionaries. For example, the script `geometry/tag_facets.py` produces the needed input files given a volume-tagged cell.

An square input mesh can be created via

```
cd geometry
python3 create_square_mesh.py
```
in *create_square_mesh.py* geometric settings (#elements and #cells) can be modified.

###  Visualize output in Paraview
In Paraview `File > Load State...` of `output/bulk_state.pvsm`, selecting the correct path in *Load State Options*, to visualise the bulk potential evolution.

TODO: membrane potential visualization

### Contributors

* Pietro Benedusi
* Edoardo Centofanti

### Cite
```
@article{benedusi2024scalable,
  title={Scalable approximation and solvers for ionic electrodiffusion in cellular geometries},
  author={Benedusi, Pietro and Ellingsrud, Ada Johanne and Herlyng, Halvor and Rognes, Marie E},
  journal={SIAM Journal on Scientific Computing},
  volume={46},
  number={5},
  pages={B725--B751},
  year={2024},
  publisher={SIAM}
}

@article{benedusi2024modeling,
  title={Modeling excitable cells with the EMI equations: spectral analysis and iterative solution strategy},
  author={Benedusi, Pietro and Ferrari, Paola and Rognes, Marie E and Serra-Capizzano, Stefano},
  journal={Journal of Scientific Computing},
  volume={98},
  number={3},
  pages={58},
  year={2024},
  publisher={Springer}
}
```


