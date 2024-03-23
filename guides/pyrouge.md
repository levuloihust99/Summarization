# Install Pyrouge
Source: https://stackoverflow.com/questions/45894212/installing-pyrouge-gets-error-in-ubuntu
## Step 1 : Install Pyrouge from source (not from pip)
```
git clone https://github.com/bheinzerling/pyrouge
cd pyrouge
pip install -e .
```
## Step 2 : Install official ROUGE script
```
git clone https://github.com/andersjo/pyrouge.git rouge
```
## Step 3 : Point Pyrouge to official rouge script
```
pyrouge_set_rouge_path ~/pyrouge/rouge/tools/ROUGE-1.5.5/
```
*The path given to pyrouge should be absolute path !*
## Step 4 : Install libxml parser
As mentioned in [this issue](https://github.com/bheinzerling/pyrouge/issues/8), you need to install libxml parser :
```
sudo apt-get install libxml-parser-perl
```
## Step 5 : Regenerate the Exceptions DB
As mentioned in [this issue](https://github.com/bheinzerling/pyrouge/issues/8), you need to regenerate the Exceptions DB :
```
cd rouge/tools/ROUGE-1.5.5/data
rm WordNet-2.0.exc.db
./WordNet-2.0-Exceptions/buildExeptionDB.pl ./WordNet-2.0-Exceptions ./smart_common_words.txt ./WordNet-2.0.exc.db
```
## Step 6 : Run the tests
```
python -m pyrouge.test
```

> Ran 11 tests in 6.322s
OK

# Install compare-mt
```shell
git clone https://github.com/neulab/compare-mt.git
cd ./compare-mt
pip install -r requirements.txt
python setup.py install
```