.PHONY: all clean

all: nitrog3d
	@:

nitrog3d:
	mkdir -p io_scene_g3d
	cp __init__.py import_nsbmd.py utils.py io_scene_g3d
	zip -r nitrog3d.zip io_scene_g3d
	rm -rf io_scene_g3d

clean:
	rm -f nitrog3d.zip
