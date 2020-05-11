IP=35.204.12.177

zip -r quantization.zip . -x *.git* -x *.idea* -x *.DS_Store* -x *__pycache__/* -x __pycache__/
scp quantization.zip ruben.steendam@$IP:~/
ssh ruben.steendam@$IP "rm -rf quantization/*; mv quantization.zip quantization/ && cd quantization && unzip -o quantization.zip && rm quantization.zip"
rm quantization.zip