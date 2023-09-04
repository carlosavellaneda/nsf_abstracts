# NFS Abstracts

This repo contains two simple approaches to identify topics in the NFS abstracts dataset. The first approach is using embeddings of a pre-trained model and then building a cluster process. The second one is using a Latent Dirichlet Allocation model. Each approach has its advantages and disadvantages, and one can even create a pipeline mixing both alternatives. The main analysis is done in a Jupyter notebook called `abstracts_clusters.ipynb`.

## Requirements

The requirements were built using poetry. To install them, run:

```bash
poetry install
```

The preprocessing steps were wrapped in classes following the scikit-learn API. The embeddings model was also wrapped in a class following the scikit-learn API. This way, it is possible to use the models in a pipeline and it is quite easy to add more steps in the pipeline.

The data was downloaded from [this link](https://www.nsf.gov/awardsearch/download?DownloadFileName=2020&All=true) and extracted on a directory called `data`.
