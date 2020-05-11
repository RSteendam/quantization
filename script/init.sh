pip3 install --user kaggle
export PATH="~/.local/bin:$PATH"
mkdir .kaggle
mv quantization/utils/kaggle.json .kaggle/
chmod 600 .kaggle/kaggle.json
mkdir data
cd data
kaggle competitions download -c dogs-vs-cats
unzip dogs-vs-cats.zip
unzip train.zip