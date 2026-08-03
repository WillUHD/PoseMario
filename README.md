<!-- 677777777777 -->

<img align="right" width="224" height="224" alt="PoseMario" src="https://github.com/user-attachments/assets/ffce99cf-cdfe-4399-9f0b-dc9eae1b3cf4" />

<h1>PoseMario</h1>

![python](https://img.shields.io/badge/python3.11-ffffff?logo=python)
![mediapipe](https://img.shields.io/badge/mediapipe-011e3d?logo=mediapipe)
![opencv](https://img.shields.io/badge/opencv-blue?logo=opencv)
![mac](https://img.shields.io/badge/macOS-000000?logo=apple)

---

### Installing `mari0_ae.app`
An efficient classic SMB runtime that works on both architectures.

1. Unzip the app either in Finder or by executing this command: 
   ```
   unzip ./install/mari0_ae.zip
   ```

2. The app should be runnable (by Right Click > `Open`, or by disabling gatekeeper). 

   > If it's not runnable, download it from the original source [here](https://www.dropbox.com/scl/fi/94eehbjlx8x4p8xc2regz/alesan99s_entities_13.2_macos.zip?rlkey=hnb24ajfjuuf3mb37c8sgbdsv&e=1&st=ftjzlc3o&dl=).

3. This app is modded with a portal gun. Unless you plan on playing like that, we have to change it back by enabling cheats: 
   ```
   chmod +x ./install/install.command && ./install/install.command
   ```
4. Verify that you can indeed use cheats in the app by going to `OPTIONS` and scrolling to `CHEATS` in the menu.

   If you want PoseMario to be playable (in my opinion), switch the mode to `CLASSIC` and turn on `INFINITE TIME` and `INFINITE LIVES`. 

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

### Previous versions
Visit [here](https://github.com/willuhd/PoseMario/tree/extras) for previous versions of PoseMario, written in Java
