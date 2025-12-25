<!-- english or spanish -->

<img align="right" width="224" height="224" alt="PoseMario" src="https://github.com/user-attachments/assets/ffce99cf-cdfe-4399-9f0b-dc9eae1b3cf4" />

<h1>PoseMario</h1>

![python](https://img.shields.io/badge/python-ffffff?logo=python)
![mediapipe](https://img.shields.io/badge/mediapipe-011e3d?logo=mediapipe)
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
  
  This is the only app I found that runs the classic SMB on Mac while being efficient. Several other apps either run on Vulkan and destroy the GPU, or only work on Apple silicon. 
  
  1. Unzip the app either in Finder or by executing this command: 
      ```
      unzip ./mari0_ae.zip
      ```
  
  2. The app should be runnable (by Right Click > `Open`, or by disabling gatekeeper). 
  
     > If it's not runnable, download it from the original source [here](https://www.dropbox.com/scl/fi/94eehbjlx8x4p8xc2regz/alesan99s_entities_13.2_macos.zip?rlkey=hnb24ajfjuuf3mb37c8sgbdsv&e=1&st=ftjzlc3o&dl=).
  
  4. Make sure the title is `Mari0: AE` and it says `MOD BY ALESAN99` at the bottom right once run.
  5. This app is modded with a portal gun. Unless you plan on playing like that, we have to change it back by enabling cheats. 
      
      1. Check if the options file exists in the first place through this command:
         ```
         [ -f ~/Library/Application\ Support/mari0/alesans_entities/options.txt ] && echo "File exists" || echo "Run mari0_ae.app"
         ```
         (If it doesn't exist, you have to run the app first)
      2. Change the contents of the options by running this command (this uses PoseMario's keybinds and enables cheats)
         ```
         echo -n "playercontrols:1:aimx-,aimy-,portal1-,left-a,up-w,right-d,down-s,run-r,pause-,jump- ,portal2-,reload-z,use-x;scale:2;letterbox:true;volume:0;vsync;gamefinished;reachedworlds:smb:1,1,1,1,1,1,1,1;resizable:true;language:english;" > ~/Library/Application\ Support/mari0/alesans_entities/options.txt
         ```
  6. Verify that you can indeed use cheats in the app by going to `OPTIONS` and scrolling to `CHEATS` in the menu.
  
     Every time you run the app, if you want PoseMario to be "playable" (in my opinion), you have to switch the mode to `CLASSIC` and turn on `INFINITE TIME` and `INFINITE LIVES`. Playing Mario with pose isn't easy!
  
      <img width="400" height="244" alt="image" src="https://github.com/user-attachments/assets/50cbdb38-e613-4148-9fc4-d0034f5efb46" />
  
      > use the arrow keys to navigate the menu

### Running the code
  
  - You may have to enable accessibility settings for your IDE (eg. Code) because PyAutoGUI has to use them to control the keyboard. 
  - Run the app via `python ./MarioController2.py`
  - Seeing logs from MediaPipe before the UI opens is normal. 
    - Press `p` to pause the app, and `p` again to resume it in 3 seconds. This is useful when setting up mari0_ae without keys pressing accidentally. 
    - Press `q` to quit the app. 
  - Note that having the UI not respond to `p`, `q`, or a resize is just a common issue from opencv's imshow. If you're really impatient, you can just `^C` in the terminal or force quit Python.
  - The app does come with a watermark by default! You can delete the watermark (`rm ./watermark.png`) or replace it with your own. 

> made with ❤️ by willuhd
