clean:
	rm -rf out/*/fold*/checkpoints/*_full.pth
	rm -rf out/*/fold*/checkpoints/last.pth
	rm -rf out/*/fold*/checkpoints/train*

jupyter:
	jupyter lab --ip=0.0.0.0 --no-browser --NotebookApp.token=""
