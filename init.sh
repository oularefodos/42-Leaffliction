python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

curl -L -o leafs.zip https://cdn.intra.42.fr/document/document/43409/leaves.zip
unzip leafs.zip
rm -rf leafs.zip
