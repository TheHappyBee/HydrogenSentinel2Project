# Hydrogen Sentinel Project

The **Hydrogen Sentinel Project** analyzes mineralogical and spectral indicators relevant to hydrogen-related geologic processes using **Sentinel-2 multispectral imagery**. This project was conducted with significant assistance from **Dr. Jiajia Sun (University of Houston)**.

---

## Repository Structure

### Code

All source files are located in the [`sider/`](./sider) directory.

#### [`integrator.py`](./sider/integrator.py)

**Description:** Core processing pipeline for the project.

* Extracts spectral band data from Sentinel-2 imagery
* Performs data preprocessing and normalization
* Conducts spectral analysis
* Outputs processed results to data files and CSVs

**Prerequisites:**

* Associated USGS datasets (e.g., `S07SNTL2`)

---

#### [`scatterplot.py`](./sider/scatterplot.py)

**Description:** Generates mineral image visualizations.

**Prerequisites:**

* [`integrator.py`](./sider/integrator.py) must be run first
* An example output is already included in this repository

---

#### [`barplot.py`](./sider/barplot.py)

**Description:** Generates Figure 1 used in the research poster.

**Prerequisites:**

* [`integrator.py`](./sider/integrator.py) must be run first

---

#### [`cluster.py`](./sider/cluster.py)

**Description:** Generates spectral clustering visualizations.

**Prerequisites:**

* [`integrator.py`](./sider/integrator.py) must be run first

---

## Images

The following images were used in the research poster and are fundamental to this work:

* [`Olivine.png`](./sider/Olivine.png)
* [`Brucite.png`](./sider/Brucite.png)
* [`Cummingtonite.png`](./sider/Cummingtonite.png)
* [`Serpentine.png`](./sider/Serpentine.png)
* [`Figure_1.png`](./sider/Figure_1.png)
* [`spectrastuff.png`](./sider/spectrastuff.png)

---

## Notes

* All paths use relative links for compatibility with GitHub and local clones.
* `integrator.py` must be executed before running any visualization scripts.
* All figures can be reproduced using the provided pipeline and datasets.
