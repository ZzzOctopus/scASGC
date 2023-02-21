# scASGC

scASGC: an adaptive simplified graph convolution model for clustering single-cell RNA-seq data

## Requirements

Python --- 3.9

Numpy --- 1.22.3

Pandas --- 1.4.1

Scipy --- 1.8.0

Sklearn --- 1.0.2

## Datasets

Deng, Yan, Leng, Darmanis were obtained from https://sourceforge.net/projects/transcriptomeassembly/files/SC3-e/Data/

Muraro and Baron-mouse were obtained from https://sourceforge.net/projects/transcriptomeassembly/files/SD-h/Data/

Buettner, Kolodziejczyk, Pollen and Zeisel were obtained from https://github.com/BatzoglouLabSU/SIMLR/tree/SIMLR/data

Biase, Goolam and Ting were obtained from https://github.com/shaoqiangzhang/scRNAseq_Datasets.

Klein and PBMC4k were obtained from https://github.com/ttgump/scDeepCluster/tree/master/scRNA-seq%20data

Chen was obtained from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE87544.

## Example

The scASGC.py is an example of the dataset Kolodziejczyk.

Input: data.csv( Rows represent cells, columns represent genes ), n_clusters( the number of cluster ), true_labels.csv

Output: pre_labels, NMI, ARI, homogeneity and completeness.

## pca+kmeans.py
This is the code for the comparison method pca+kmeans.


## The specifications of the computer used for data analysis and comparison
The dataset chen was run on a Tesla V100 and all other datasets were run on a personal computer (i5-8250U,64bit,windows10).
