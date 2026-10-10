# 📊 iTrust SWaT Dataset Preprocess Tool

This is a tool for iTrust SWaT dataset preprocessing, including dataset details introduction, dataset visulization, dataset processing and some pytorch utils.

> **Status in this repo:** standalone EDA/visualizer code (see
> `data/README.md` for how `data/` is populated here). It is separate from
> `src/data.py`'s `load_swat()`, which is what the actual experiment
> pipeline trains and evaluates on — nothing here feeds back into that
> pipeline. Two deviations made so this runs against the real 2015 attack
> data rather than a hand-transcribed subset:
> - `notebook/swat-2015-info.ipynb` derives attack windows straight from
>   the dataset's own `Normal/Attack` column instead of a manually
>   typed-in list for one day, so it shades every real attack window
>   (all 51 sensors, since the row label alone doesn't say which sensor
>   was targeted) rather than one hand-picked sensor per window.
> - The 51-panel plots (`fig/swat-2015.png`, `fig/swat-2015-sr.png`) are
>   built at 60x100in / 6000x10000px, saved via `plt.savefig()` and
>   explicitly `plt.show()`-n — open the PNG directly if a notebook
>   viewer doesn't render an embedded image that large.
>
> **Reading the plots:** the real data covers `2015-12-28 10:00:00` →
> `2016-01-02 14:59:59` (449,919 rows, one per second, essentially
> gapless). Every plot here only shows a single 24-hour slice of that —
> `29 Dec 2015` by default, set by `time_start`/`time_end` in each
> notebook — not the full ~5-day range; change those two lines to look
> at a different day. Each row in the figure is one sensor/actuator; the
> shaded red bands are real attack windows (from the `Normal/Attack`
> column) that fall inside the plotted day.

## 💡 Introduction

iTrust Secure Water Treatment (SWaT) dataset is a famous and widely used dataset, mostly in anomaly detection field. For more details about this dataset, please refer to the official site for SWaT dataset: [https://itrust.sutd.edu.sg/testbeds/secure-water-treatment-swat/](https://itrust.sutd.edu.sg/testbeds/secure-water-treatment-swat/)

Since SWaT is originally collected dataset, by using it, we should do some preprocessing. Here we provide a tool to help you do this. Here we mostly forcus on SWaT 2015 dataset since it provides more clear attack informations. About other years data, we will keep updating. Here are features about our tool. 

Features: 

- Data visulization: we visulized the dataset, with marking anomaly parts, which is clear to check data patterns. 
- Excel tools: we provide some tools to help you read excel elegantly, like reading attack lists and return attacks within what period you choose. 
- Spectral residual preprocessing: since the SWaT dataset has obvious cycle pattern, we apply spectral residual method to help us remove redundancy. 
- Pytorch dataset class: we provide pytorch dataset class, can be applied in time-series anomaly detection trainning. 

## 🔧 Environment 

``` 
Python Version: 3.6 or later
Python Packages: jupyterlab, torch, numpy, pandas, matplotlib, sklearn
```

## 📁 Structure

```
.
├── data/
│   ├── swat-2015-data.csv
│   ├── swat-2015-data.npy
│   └── swat-2015-attack.xlsx
├── fig
├── notebook/
│   ├── swat-2015-info.ipynb
│   ├── swat-2015-process.ipynb
│   └── swat-2015-dataset.ipynb
├── src/
│   ├── silency.py
│   └── util.py
└── README.md
```

- `data/`: all dataset files will be here
    - `swat-2015-data.csv`: original dataset, transferred from xlsx file
    - `swat-2015-data.npy`: transferred dataset
    - `swat-2015-attack.xlsx`: original attack information
- `fig/`: result figures will be here
- `notebook/`: preprocess in notebook format
    - `swat-2015-info.ipynb`: introduction about the dataset, including basic information and visulization
    - `swat-2015-process.ipynb`: main preprocess on the dataset, including normalization and SR method
    - `swat-2015-dataset.ipynb`: about using the torch dataset class tool
- `script/`: preprocess in script format
- `src/`: source code of tools 

## 📚 References

[1]. Hou, Xiaodi, and Liqing Zhang. **"Saliency detection: A spectral residual approach."** *2007 IEEE Conference on computer vision and pattern recognition.* Ieee, 2007.
