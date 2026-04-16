.PHONY: clean-docker clean-docker-all docker-usage

clean-docker:
	@echo "Pruning dangling images and build cache (keeping 5GB cache, volumes untouched)..."
	docker image prune -f
	docker builder prune -f --keep-storage 5GB
	@echo "Done. Volumes and Ollama models preserved."

clean-docker-all:
	@echo "Aggressive prune: dangling images + ALL build cache. Volumes and Ollama models preserved."
	docker image prune -f
	docker builder prune -af
	@echo "Done."

docker-usage:
	docker system df
