# autocorrstats

<img width="989" height="396" alt="image" src="https://github.com/user-attachments/assets/4f1e2571-9675-49f1-977e-3d61dbbfa191" />

`autocorrstats` helps you test relationships in autocorrelated time series without relying on methods that assume independent samples. It follows Ebisuzaki (1997) to generate ensembles of synthetic time series with power spectra similar to those of the original data, allowing you to estimate the significance of statistical quantities.

In this repository, the word `surrogates` refers to those synthetic time series. They are not new observations; they are randomized series generated from the original data so that key properties such as the power spectrum are preserved while the timing information is scrambled.

The package currently supports significance testing for correlations and polynomial fits in time series that exhibit autocorrelation.

## Installation

The simplest setup is the included Conda environment:

```bash
conda env create -f environment.yml
conda activate autocorrstats
```

This installs the package in editable mode together with the development and notebook dependencies used in the examples.

If you only want to install the package itself:

```bash
pip install git+https://github.com/anthony-meza/SpectralCorr.git@main
```

## License

`autocorrstats` was created by Anthony Meza and is released under the MIT License.

## Reference

Ebisuzaki, W. (1997). A method to estimate the statistical significance of a correlation when the data are serially correlated. Journal of Climate, 10(9), 2147-2153. [https://doi.org/10.1175/1520-0442(1997)010%3C2147:AMTETS%3E2.0.CO;2](https://doi.org/10.1175/1520-0442(1997)010%3C2147:AMTETS%3E2.0.CO;2)
