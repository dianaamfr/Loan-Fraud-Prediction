# clean PARAM='<output_file>'
# make clean PARAMS=sub2
clean:
	python clean.py ${PARAMS}

# train PARAM='<classifier> <clean_data> <output_name>'
# make train PARAMS='logistic_regression sub2 sub2'
train:
	python train.py ${PARAMS}

# test PARAM='<classifier> <clean_data> <output_name>'
# make test PARAMS='logistic_regression sub2 sub2'
test:
	python test.py ${PARAMS}

clean_models:
	rm -rf models/*

clean_cache:
	rm -rf __pycache__
