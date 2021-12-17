# Evaluation of DP Mobility Report

This ist the code to run the evaluation as stated in the publication xx.

Output for all error measures can be found [here as tables](results/tables) and [here as graphs](results/graphs/graphs_all_error_measures.pdf).

## Reproduce results

The evaluation uses Version 0.0.1 of the [`dp_mobility_report`](https://github.com/FreeMoveProject/dp_mobility_report) package.

To reproduce the results, follow these instructions.

Create a conda environment:

``` bash
conda create --name eval_dpmr
conda activate eval_dpmr
pip install -r requirements.txt
```

Within `config.py` you can define data sets you can to include in the evaluation. As `BERLIN` is not openly available, you might want to exclude it. You can also either create this synthetic data yourself, using [TAPAS](https://github.com/DLR-VF/TAPAS) or contact us to provide the data.

The evaluation consists over four substeps:

1. **Preprocess the input data:** [GEOLIFE](https://www.microsoft.com/en-us/download/details.aspx?id=52367) and [MADRID](https://crtm.maps.arcgis.com/apps/MinimalGallery/index.html?appid=a60bb2f0142b440eadee1a69a11693fc) are open data sets. 'python 01_preprocess_evaluation_data.py' will download and preprocess those. It also includes preprocessing for BERLIN, though this data is expected to be present end 'data/raw/berlin'.
2. **Compute reports:** The seconds scripts computes reports of all `max_trips M` and `privacy_budget eps` combinations with 10 repetitions. This results in 480 runs for GEOLIFE, 360 runs for MADRID and 420 runs for BERLIN (the difference is because of different numbers of `max_trips` variations). One run for GEOLIFE takes about 30 seconds, 1 minute for MADRID and 3 minutes for BERLIN. Thus, this step takes roughly a total of about 4 hours + 6 hours + 21 hours = 31 hours!
3. **Compute error measures:** Based on the precomputed reports, error measures are calculated and stored in results as csv files.
4. **Plotting of results:** One output specifically plots results as presented in the paper. Another output plots results for all computed error measures (also those not stated in the publication for reasons of brevity).

To run all steps, execute:

``` bash
bash 00_evaluation.bash
```

