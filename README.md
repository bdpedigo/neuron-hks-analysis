# neuron-hks-analysis

Scripts and aggregated data associated with [Pedigo et al. 2026 - A quantitative census of millions of postsynaptic structures in a large electron microscopy volume of mouse visual cortex](https://www.biorxiv.org/content/10.64898/2026.02.19.706834v2)

## Installation

Clone this repository: `git clone https://github.com/bdpedigo/neuron-hks.git`

We use [`uv`](https://docs.astral.sh/uv/) for dependency management (note that this does not preclude the use of `pip` if you prefer).

Dependencies are listed in `pyproject.toml`. Create a new environment and install all of them by navigating to this repo and then running `uv sync`.

You may also need to set up CAVEclient access if you have not used MICrONS on your 
computer. Follow the instructions [here](https://tutorial.microns-explorer.org/quickstart_notebooks/01-caveclient-setup.html) (this should only need to happen once).

## Running the code

Currently there is only one mega-script in `./scripts`. Running it will generate many
figure panels and save output variables. To run, do `uv run ./scripts/stat_summaries.py`.

## Contact

If you have any questions or comments about this code, feel free to reach out: [ben.pedigo@alleninstitute.org](mailto:ben.pedigo@alleninstitute.org).