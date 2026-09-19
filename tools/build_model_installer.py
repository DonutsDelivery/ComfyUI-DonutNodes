"""Build the separate Civitai installer ZIP using the current shared catalog."""
import argparse
from pathlib import Path
import zipfile


def build(destination):
    root = Path(__file__).resolve().parents[1]
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in ('install_models.py', 'install-models.bat', 'install-models.sh', 'README.md'):
            archive.write(root / 'model-installer' / name, 'model-installer/' + name)
        archive.write(root / 'model_sources.json', 'model-installer/model_sources.json')
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    print(build(parser.parse_args().destination))
