IP=34.90.102.0

zip -r quantization.zip . -x *.git* -x *.idea* -x *.DS_Store* -x *__pycache__/* -x __pycache__/
scp quantization.zip ruben.steendam@$IP:~/
ssh ruben.steendam@$IP "rm -rf quantization/*; mv quantization.zip quantization/ && cd quantization && unzip -o quantization.zip && rm quantization.zip"
rm quantization.zip