# CMR-Sum
This is the source code for "Cross-Modal Retrieval-Enhanced Code Summarization based on Joint Learning for Retrieval and Generation"

## Package Requirement

To run this code, some packages are needed as following:

```python
torch==2.0.0
torchvision==0.15.0
numpy==1.23.5
tqdm==4.65.0
transformers==4.27.1
nltk==3.8.1
faiss==1.7.2
```
Moreover, a jdk is also needed to run the evaluation script for METEOR.

## Dataset
Prepare dataset for java and python:
```
cd dataset
unzip data.zip
```

## Training and testing
For the Java dataset:
```
bash run_java.sh
```
For Python data:
```
bash run_python.sh
```
The model checkpoints, generated results, retrieved results and reference will be saved in the "saved_models" folder. 



