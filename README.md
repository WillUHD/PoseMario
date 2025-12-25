<!-- english or spanish -->

<img align="right" width="224" height="224" alt="PoseMario" src="https://github.com/user-attachments/assets/ffce99cf-cdfe-4399-9f0b-dc9eae1b3cf4" />

<h1>PoseMario</h1>

![python](https://img.shields.io/badge/python3.11-ffffff?logo=python)
![mediapipe](https://img.shields.io/badge/mediapipe-011e3d?logo=mediapipe)
![opencv](https://img.shields.io/badge/opencv-blue?logo=opencv)
![mac](https://img.shields.io/badge/macOS-000000?logo=apple)
![deprecated](https://img.shields.io/badge/⚠️DEPRECATED!-secretMessage!!!?color=orange)

> [!warning]
> This branch is **deprecated** and no longer actively maintained!!
> 
> (it  works, but I won't be touching it, so feel free to pr if you have a fix)

(make sure you've `cd`'d into the src directory!)

---

### Dependencies

Execute this command to install from requirements: 
```
pip install -r ./requirements.txt
```

...or, install everything by executing this: 
```
pip install 'absl-py==2.3.1' 'attrs==25.4.0' 'certifi==2025.11.12' 'cffi==2.0.0' 'charset-normalizer==3.4.4' 'colorama==0.4.6' 'coloredlogs==15.0.1' 'contourpy==1.3.3' 'cycler==0.12.1' 'filelock==3.20.1' 'flatbuffers==25.9.23' 'fonttools==4.61.1' 'fsspec==2025.12.0' 'humanfriendly==10.0' 'idna==3.11' 'jax==0.4.38' 'jaxlib==0.4.38' 'Jinja2==3.1.6' 'kiwisolver==1.4.9' 'MarkupSafe==3.0.3' 'matplotlib==3.10.8' 'mediapipe==0.10.21' 'ml_dtypes==0.5.4' 'MouseInfo==0.1.3' 'mpmath==1.3.0' 'networkx==3.6.1' 'numpy==1.26.4' 'onnx==1.20.0' 'onnxruntime==1.19.2' 'onnxslim==0.1.80' 'opencv-contrib-python==4.11.0.86' 'opencv-python==4.11.0.86' 'opt_einsum==3.4.0' 'packaging==25.0' 'pillow==12.0.0' 'polars==1.36.1' 'polars-runtime-32==1.36.1' 'protobuf==4.25.8' 'psutil==7.1.3' 'PyAutoGUI==0.9.54' 'pycparser==2.23' 'PyGetWindow==0.0.9' 'PyMsgBox==2.0.1' 'pyobjc-core==12.1' 'pyobjc-framework-Cocoa==12.1' 'pyobjc-framework-Quartz==12.1' 'pyparsing==3.2.5' 'pyperclip==1.11.0' 'PyRect==0.2.0' 'PyScreeze==1.0.1' 'python-dateutil==2.9.0.post0' 'pytweening==1.2.0' 'PyYAML==6.0.3' 'requests==2.32.5' 'rubicon-objc==0.5.3' 'scipy==1.16.3' 'sentencepiece==0.2.1' 'six==1.17.0' 'sounddevice==0.5.3' 'sympy==1.14.0' 'torch==2.2.2' 'torchvision==0.17.2' 'tqdm==4.67.1' 'typing_extensions==4.15.0' 'urllib3==2.6.2'
```

### Installing `mari0_ae.app`
  
Refer to [this link](https://github.com/WillUHD/PoseMario?tab=readme-ov-file#installing-mari0_aeapp) in the `main` branch for the installation instructions! 

### Running the code
  
  - You may have to enable accessibility settings for your IDE (eg. Code) because PyAutoGUI has to use them to control the keyboard. 
  - Run the app via `python ./MarioController2.py`
  - Seeing logs from MediaPipe before the UI opens is normal. 
    - Press `p` to pause the app, and `p` again to resume it in 3 seconds. This is useful when setting up mari0_ae without keys pressing accidentally. 
    - Press `q` to quit the app. 
  - Note that having the UI not respond to `p`, `q`, or a resize is just a common issue from opencv's imshow. If you're really impatient, you can just `^C` in the terminal or force quit Python.
  - The app does come with a watermark by default! You can delete the watermark (`rm ./watermark.png`) or replace it with your own. 

> made with ❤️ by willuhd
