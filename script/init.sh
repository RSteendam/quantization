pip install --user kaggle
export PATH="~$HOME/.local/bin:$PATH"
mkdir .kaggle
mv kaggle.json .kaggle/
kaggle competitions download -c dogs-vs-cats
unzip dogs-vs-cats.zip
unzip train.zip