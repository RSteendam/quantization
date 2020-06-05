IP=###
USER=###

zip -r quantization.zip . -x *.git* -x *.idea* -x *.DS_Store* -x *__pycache__/* -x __pycache__/
scp quantization.zip $USER@$IP:~/
ssh $USER@$IP "mkdir quantization; mv quantization.zip quantization/ && cd quantization && unzip -o quantization.zip && rm quantization.zip"
rm quantization.zip
ssh $USER@$IP "sh quantization/script/init.sh"