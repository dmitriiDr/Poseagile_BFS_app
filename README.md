🧘 Yoga Pose Detector – Test Release v0.1
We’re excited to share the first test version of our yoga pose detection app!

🔍 Key Features:

📸 Real-time pose detection using your device’s camera

⏱️ Automatic tracking of how long you hold each yoga pose

🧍 Visual feedback to help you align and improve your posture

🧪 This is a test release, so please note:

Some poses may not be recognized perfectly

Timer accuracy may vary depending on lighting and camera angle

User interface is minimal and still under development

🙏 We’d love your feedback to improve the app!
If you notice bugs or have suggestions, please let us know.

# Install poetry
```bash
curl -sSL https://install.python-poetry.org | python3
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Initialize environment 
poetry install
```

# with a video already recorded
```bash
poetry run python yoga_app/yoga.py --video input.mp4 --output output_yoga.mp4
```

# without a video recorded (webcam is required)
```bash
poetry run python yoga_app/yoga.py --output output_yoga.mp4
```