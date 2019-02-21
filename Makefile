isort:
	pipenv run isort -ns __init__.py -rc -p gym_agents gym_agents

format:
	pipenv run autopep8 gym_agents -r --in-place
