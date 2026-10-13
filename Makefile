PY=app/venv/bin/python
.PHONY: venv assets test local lint
venv:
	python3 -m venv app/venv && app/venv/bin/pip install -U pip && app/venv/bin/pip install -r app/requirements.txt
assets:
	@test -f app/ignored/menagerie/wonik_allegro/right_hand.xml || ( git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie app/ignored/menagerie && cd app/ignored/menagerie && git sparse-checkout set wonik_allegro )
	$(PY) app/build_cell.py
test:
	$(PY) -m pytest -q app/tests
local:
	cd app && venv/bin/python run-decision-hand.py --local --stage lift --max-steps 400 --run-tag local
lint:
	$(PY) -m py_compile app/*.py
