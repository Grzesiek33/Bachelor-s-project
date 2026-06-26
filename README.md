# Bachelor's project

This repository contains code for working with satellite image geolocation models, Ground Control Points (GCPs), and camera model correction/optimization. The project focuses on three supported study areas:

- `San_francisco`
- `Angkor_wat`
- `Cocabamba`

The code includes utilities for parsing metadata, building PSM and RFM models, optimizing correction parameters, and visualizing accuracy results.

## Repository structure

- `src/utils/` — helper modules, parsers, and camera/model utilities
- `src/optimize/` — optimization routines
- `src/showcase/` — scripts for plots, tables, and visual results
- `San_francisco/`, `Angkor_wat/`, `Cochabamba/` — project data for each city
- `optimization/` — saved optimization outputs

## Requirements

The project is written in Python and uses libraries such as:

- NumPy
- PyTorch
- SciPy
- Matplotlib
- Rasterio
- Shapely
- SymPy

If needed, install the dependencies in your environment before running the scripts.

## Data preparation

Before starting the project, you must download the required data for each city and place it in the corresponding city folder.

### Required files

For each supported city, make sure the following data are available:

- `l1a_frames/`
- `frame_index.csv`

### Download sources

Use the following links to obtain the data:

- **San Francisco**  
  Download from:  
  https://www.planet.com/data/stac/browser/open-skysat-data/san-francisco/20210101_184738_ssc13_u0001/20210101_184738_ssc13_u0001.json?.language=en&.asset=asset-all-frames

- **Angkor Wat**  
  Download from:  
  https://www.planet.com/data/stac/browser/open-skysat-data/angkor-wat/20201214_032156_ssc3_u0001/20201214_032156_ssc3_u0001.json?.language=en&.asset=asset-visual%3Aortho_visual

- **Cocabamba**  
  Download from:  
  https://www.planet.com/data/stac/browser/open-skysat-data/cocabamba-peru/20201230_151832_ssc13_u0001/20201230_151832_ssc13_u0001.json?.language=en&.asset=asset-all-frames

### Folder placement

After downloading, place the data inside the corresponding city directory:

```text
San_francisco/
├── frame_index.csv
├── l1a_frames/
└── ...

Angkor_wat/
├── frame_index.csv
├── l1a_frames/
└── ...

Cocabamba/
├── frame_index.csv
├── l1a_frames/
└── ...

```

### Important preprocessing step

Before running any optimization or evaluation scripts, run:

```bash
python src/utils/RPC_parser.py
```

This script parses the RPC .TXT files in the l1a_frames/ directory and generates frameRPC.json for the selected city.

### Typical workflow

1. Download the required data for each city.
2. Place `l1a_frames/` and `frame_index.csv` into the correct city folder.
3. Run `src/utils/RPC_parser.py` to generate RPC metadata.
4. Prepare or update the GCP files in `own_GCPs/`.
5. Run optimization scripts from `src/optimize/`.
6. Generate images, plots, and tables from `src/showcase/`.

### Example usage

See the `examples/` folder for example scripts demonstrating how to use the provided utilities and optimization routines.

### Notes

- The project assumes the existence of city-specific data files in the expected directory structure.
- Some scripts use relative paths, so they should be run from the project root.
- The supported city list is defined in `src/utils/cities.py`.