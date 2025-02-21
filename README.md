环境为python3.7
pythonocc 7.5.1
occwl 3.0
flask 2.2.2
numpy 1.12.5

1.创建环境
conda create --name stepParsing python=3.7
y

conda activate stepParsing

pip install numpy
y

conda install Flask
y

conda install -c conda-forge pythonocc-core=7.5.1
y

conda install -c lambouj -c conda-forge occwl
y

2.运行服务器
cd stepParsing

python -m flask run


