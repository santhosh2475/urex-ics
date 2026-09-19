# Place for dataset

Based on iTrust's license:

`Not share the dataset with others, whether in a private or public setting (all additional requests for the same dataset needs to go through the request form).`

We can not directly share dataset here. So please refer the offical site of iTrust SWaT dataset to apply this dataset (it will be pretty easy to apply but need one or two days to wait for their reply).

iTrust Dataset Officlal Website: [https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/).

## In this repo

This directory is gitignored (same iTrust redistribution restriction as
`data/raw/`) and regenerated locally, not checked in. It's built from the
same real download the main project's `load_swat()` uses
(`data/raw/swat/SWaT_Dataset_Attack_v0.xlsx`), converted to what these
notebooks expect with:

```python
import pandas as pd
pd.read_excel(
    "../../data/raw/swat/SWaT_Dataset_Attack_v0.xlsx", header=1
).to_csv("swat-2015-data.csv", index=False)
```

Run `notebook/swat-2015-process.ipynb` after that to produce
`swat-2015-data.npy` from the csv. Both files are ~130-180MB; regenerate
them with the snippet above rather than expecting them to persist.
