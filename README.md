<!-- 677777777777 -->

<img align="right" width="224" height="224" alt="PoseMario" src="https://github.com/user-attachments/assets/ffce99cf-cdfe-4399-9f0b-dc9eae1b3cf4" />

<h1>PoseMario</h1>

![java25](https://img.shields.io/badge/java25-ED8B00?&logo=openjdk&logoColor=white)
![maven](https://img.shields.io/badge/mvn-4fa4ff?logo=apachemaven)
![onnxruntime](https://img.shields.io/badge/onnxruntime-3d251e?logo=onnx)
![mac](https://img.shields.io/badge/macOS-000000?logo=apple)

(this is meant to be run on JDK 25 (LTS). make sure you've `cd`'d into the src directory!)

---

### Dependencies

- Your IDE (eg. IDEA) will automatically start resolving the `pom.xml` file. If you're on the commandline, run `mvn install`
- Or, if you're manually importing, download the sources, classes, and (JavaDocs) for `com.microsoft.onnxruntime:1.18.0` and `org.openpnp.opencv:4.9.0-0`. 

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
  
  - You may have to enable accessibility settings for your IDE (eg. IDEA) because AWT Robot has to use it to control the keyboard. 
  - Seeing logs from ONNX Runtime and Java before the UI opens is normal (namely shouting about a restricted method and warning you about the CoreML fallback nodes)
  - The app does come with a watermark by default! You can delete the watermark (`rm ./watermark.png`) or replace it with your own.
  - If you encounter accuracy issues with RTM, I've also worked with YOLO11n and MoveNet previously (albeit with worse performance). The code for these are located [here](https://github.com/WillUHD/scratch/tree/PoseMario). Note that these will not be maintained ! 

> made with ❤️ by willuhd
