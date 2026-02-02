# Dataset_labelling_tool
This script is a simple OpenCV-based image annotation tool for creating YOLO-format bounding box labels. It lets you draw boxes with your mouse, assign classes using number keys, and automatically saves labels in YOLO format.

# Why
I made this tool because surpisingly and frustratingly I literally couldn't find a single dataset labelling tool that worked. Maybe I didn't search hard enough. Regardless, I made this simple tool, you just need to have your picture files in the right place, run the script and it will save the labels in a YOLO format, useful for computer vision projects. 👁️🖲️

# Setup
Your folder should look like this.
```
project/
│
├── images/          # Put all images you want to label here
├── labels/          # Created automatically
├── classes.txt      # Created automatically (editable)
├── labeler.py       # This script
```
Edit the classes file, run the script, and it will output labels.
