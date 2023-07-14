


all: 
	script/compile_all.sh

debug: 
	python ./doc/build.py

fn: 
	python3 script/get_all_functions_c.py