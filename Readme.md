## Setup

Install all in module_ai repository first.


#### installing python packages trough pip

In the root path of this repository:
	
	virtualenv -p python3 venv
	source venv/bin/activate
	pip install -r requirements.txt
	python setup.py develop

go to module_ai project and in order to run the ai module API, run:

	source venv/bin/activate
	python api/main_api.py -c ./config/config_default.json

note that these last two commands are assumed to be executed on root directory at module_ai repository.

# API

Documentation __[link](https://github.com/astng/module_ai/wiki/Module-AI-API-Documentation)__
