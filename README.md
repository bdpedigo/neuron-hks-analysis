# neuron-hks-analysis

Scripts and aggregated data associated with [Pedigo et al. 2026 - A quantitative census of millions of postsynaptic structures in a large electron microscopy volume of mouse visual cortex](https://www.biorxiv.org/content/10.64898/2026.02.19.706834v2)

## Running the code

- Clone this repository: 
  
  ```git clone https://github.com/bdpedigo/neuron-hks.git```

- Make sure you have [`uv`](https://docs.astral.sh/uv/) installed on your system for dependency management. Note that this does not preclude the use of `pip` if you prefer, but I have had better luck with reproducibility and simplicity using `uv`, so these instructions assume `uv`.
- Currently there is only one mega-script in `./scripts`. Running it will generate many figure panels and save output variables. To run, do:

    ```uv run ./scripts/stat_summaries.py```

- At last run, this script took about 7.5 minutes to run on my MacBook Pro laptop.
- The code uses Python 3.12 and should run on Mac, Linux, or Windows - I have tested this using a GitHub Actions worklow, but let me know if you run into any issues.

## Notes
- This script starts from some intermediate aggregate tables about cell inputs and outputs, as well as a selected synapse table (inputs and outputs to curated cells) with additional information attached. Generating these tables required access to some large cloud storage tables that we currently don't have a mechanism of sharing, but are working on.
- Similarly, you should not need CAVEclient access to run these scripts because I have pre-baked these tables and included them in the repo. If you would like to set up CAVEclient yourself and explore the raw tables, follow the instructions [here](https://tutorial.microns-explorer.org/quickstart_notebooks/01-caveclient-setup.html) (this should only need to happen once).

## Other Resources
- [Main website for the paper](https://bdpedigo.github.io/neuron-hks/)
- [Interactive demo of running HKS on a neuron from a cloud path or mesh file](https://bdpedigo.github.io/neuron-hks/demo/)
- [Example of running the HKS pipeline on a neuron](https://bdpedigo.github.io/meshmash/examples/tutorial/)
- [Instructions on programatic access to the raw prediction tables](https://bdpedigo.github.io/neuron-hks/programmatic-access/)
- [Example views using this data in Neuroglancer](https://bdpedigo.github.io/neuron-hks/neuroglancer/)

## Contact

If you have any questions or comments about this code, feel free to reach out: [ben.pedigo@alleninstitute.org](mailto:ben.pedigo@alleninstitute.org).