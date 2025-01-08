
.PHONY: up
up:
	docker compose up --build

.DEFAULT_GOAL := up
