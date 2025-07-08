# Install poetry
curl -sSL https://install.python-poetry.org | python3
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
poetry install

# with a video already recorded
poetry run python yoga.py --video input.mp4 --output output_yoga.mp4

# without a video recorded (webcam is required)
poetry run python yoga.py --output output_yoga.mp4