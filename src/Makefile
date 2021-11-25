# clean PARAM='<submission_name>'
# make clean PARAMS=sub2
clean:
	python clean.py ${PARAMS}

# train PARAM='<classifier> <submission_name>'
# make train PARAMS='logistic_regression sub2'
train:
	python train.py ${PARAMS}

# test PARAM='<classifier> <submission_name>'
# make test PARAMS='logistic_regression sub2'
test:
	python test.py ${PARAMS}

clean_models:
	rm -rf models/*

clean_cache:
	rm -rf __pycache__
