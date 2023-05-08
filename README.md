# INSTALLATION GUIDE
## 1. Setup virtual environment
```shell
$ python3 -m venv .venv
$ source .venv/bin/activate
(.venv)$ pip install -U pip
(.venv)$ pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
(.venv) pip install transformers==4.6.1 sentencepiece tensorboard nltk
```

## 2. Install Pyrouge
Source: https://stackoverflow.com/questions/45894212/installing-pyrouge-gets-error-in-ubuntu
### Step 1 : Install Pyrouge from source (not from pip)
<pre>
git clone https://github.com/bheinzerling/pyrouge
cd pyrouge
pip install -e .
</pre>
### Step 2 : Install official ROUGE script
<pre>
git clone https://github.com/andersjo/pyrouge.git rouge
</pre>
### Step 3 : Point Pyrouge to official rouge script
<pre>
pyrouge_set_rouge_path ~/pyrouge/rouge/tools/ROUGE-1.5.5/
</pre>
*The path given to pyrouge should be absolute path !*
### Step 4 : Install libxml parser
As mentioned in [this issue](https://github.com/bheinzerling/pyrouge/issues/8), you need to install libxml parser :
<pre>
sudo apt-get install libxml-parser-perl
</pre>
### Step 5 : Regenerate the Exceptions DB
As mentioned in [this issue](https://github.com/bheinzerling/pyrouge/issues/8), you need to regenerate the Exceptions DB :
<pre>
cd rouge/tools/ROUGE-1.5.5/data
rm WordNet-2.0.exc.db
./WordNet-2.0-Exceptions/buildExeptionDB.pl ./WordNet-2.0-Exceptions ./smart_common_words.txt ./WordNet-2.0.exc.db
</pre>
### Step 6 : Run the tests
<pre>
python -m pyrouge.test
</pre>

> Ran 11 tests in 6.322s
OK

## 3. Install compare-mt
```shell
git clone https://github.com/neulab/compare-mt.git
cd ./compare-mt
pip install -r requirements.txt
python setup.py install
```