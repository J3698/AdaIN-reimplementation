aria2c --file-allocation=none -c -x 10 -s 10 -d "./" http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart.zip
aria2c --file-allocation=none -c -x 10 -s 10 -d "./" http://images.cocodataset.org/zips/train2017.zip
aria2c --file-allocation=none -c -x 10 -s 10 -d "./" http://images.cocodataset.org/zips/val2017.zip
aria2c --file-allocation=none -c -x 10 -s 10 -d "./" http://images.cocodataset.org/annotations/annotations_trainval2017.zip

mkdir datasets
mv train2017.zip datasets
mv val2017.zip datasets
mv annotations_trainval2017.zip datasets
mv wikiart.zip datasets
cd datasets
for f in *.zip; do
	unzip $f >/dev/null
done
