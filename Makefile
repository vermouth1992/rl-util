.PHONY: build upload clean upload_test

build:
	python -m build

clean:
	rm -rf dist/

upload_test:
	twine upload --repository testpypi dist/*

upload:
	twine upload dist/*