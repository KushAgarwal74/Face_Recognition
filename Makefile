.PHONY: venv install train docker docker-run clean

venv:
	python3 -m venv .venv

install:
	. .venv/bin/activate && pip install -r requirements.txt

train:
	. .venv/bin/activate && python -m app.train

docker:
	docker build -t face-recognition -f docker/Dockerfile .

docker-run:
	docker run --rm face-recognition

clean:
	rm -rf .venv
