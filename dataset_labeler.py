import cv2
import os
import json
from pathlib import Path


class YOLOLabeler:
    def __init__(self, image_folder, output_folder="labels", classes_file="classes.txt"):
        self.image_folder = Path(image_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)

        # Load or create classes
        self.classes_file = Path(classes_file)
        self.classes = self.load_classes()
        self.current_class = 0

        self.images = sorted([f for f in self.image_folder.glob("*")
                              if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        self.current_idx = 0
        self.current_boxes = []
        self.drawing = False
        self.start_point = None

        if not self.images:
            raise ValueError(f"No images found in {image_folder}")

        print("\n=== YOLO Dataset Labeler ===")
        print("\nClasses:")
        for i, cls in enumerate(self.classes):
            print(f"  {i}: {cls}")
        print("\nControls:")
        print("  Draw box: Click and drag")
        print("  Change class: Number keys (0-9)")
        print("  Delete last box: 'z'")
        print("  Save & Next: 'n' or Space")
        print("  Previous: 'p'")
        print("  Quit: 'q' or ESC")
        print(f"\nFound {len(self.images)} images")
        print(f"Current class: [{self.current_class}] {self.classes[self.current_class]}\n")

    def load_classes(self):
        if self.classes_file.exists():
            with open(self.classes_file, 'r') as f:
                classes = [line.strip() for line in f if line.strip()]
            print(f"Loaded classes from {self.classes_file}")
            return classes
        else:
            # Create default classes file
            default_classes = ["oak", "pine", "maple", "birch", "willow"]
            with open(self.classes_file, 'w') as f:
                f.write('\n'.join(default_classes))
            print(f"Created default classes file: {self.classes_file}")
            print("Edit this file to add your own tree types!")
            return default_classes

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            img_copy = self.display_img.copy()
            color = self.get_class_color(self.current_class)
            cv2.rectangle(img_copy, self.start_point, (x, y), color, 2)
            # Show class name while drawing
            cv2.putText(img_copy, self.classes[self.current_class],
                        (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.imshow('YOLO Labeler', img_copy)

        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            x1, y1 = self.start_point
            x2, y2 = x, y

            # Ensure x1 < x2 and y1 < y2
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            # Ignore very small boxes
            if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
                self.update_display()
                return

            # Convert to YOLO format (normalized center_x, center_y, width, height)
            h, w = self.img.shape[:2]
            center_x = ((x1 + x2) / 2) / w
            center_y = ((y1 + y2) / 2) / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h

            self.current_boxes.append([self.current_class, center_x, center_y, width, height])
            print(f"Added box: {self.classes[self.current_class]}")
            self.update_display()

    def get_class_color(self, class_id):
        # Generate distinct colors for each class
        colors = [
            (0, 255, 0),  # Green
            (255, 0, 0),  # Blue
            (0, 0, 255),  # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 255, 0),  # Spring green
            (255, 128, 0),  # Azure
            (128, 0, 255),  # Purple
            (255, 165, 0),  # Orange
        ]
        return colors[class_id % len(colors)]

    def update_display(self):
        self.display_img = self.img.copy()
        h, w = self.img.shape[:2]

        # Draw existing boxes
        for box in self.current_boxes:
            class_id, cx, cy, bw, bh = box
            class_id = int(class_id)

            # Convert back to pixel coordinates
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            color = self.get_class_color(class_id)
            cv2.rectangle(self.display_img, (x1, y1), (x2, y2), color, 2)

            # Add label with class name
            label = self.classes[class_id] if class_id < len(self.classes) else f"Class {class_id}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(self.display_img, (x1, y1 - label_size[1] - 8),
                          (x1 + label_size[0] + 4, y1), color, -1)
            cv2.putText(self.display_img, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Add info panel
        info_height = 80
        info_panel = self.display_img[:info_height].copy()
        info_panel[:] = (50, 50, 50)  # Dark gray background
        self.display_img[:info_height] = cv2.addWeighted(info_panel, 0.7, self.display_img[:info_height], 0.3, 0)

        # Image info
        info = f"Image {self.current_idx + 1}/{len(self.images)} | Boxes: {len(self.current_boxes)}"
        cv2.putText(self.display_img, info, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Current class
        class_color = self.get_class_color(self.current_class)
        class_text = f"Current: [{self.current_class}] {self.classes[self.current_class]}"
        cv2.putText(self.display_img, class_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, class_color, 2)

        cv2.imshow('YOLO Labeler', self.display_img)

    def save_labels(self):
        img_path = self.images[self.current_idx]
        label_path = self.output_folder / f"{img_path.stem}.txt"

        with open(label_path, 'w') as f:
            for box in self.current_boxes:
                f.write(' '.join(map(str, box)) + '\n')

        print(f"Saved {len(self.current_boxes)} boxes to {label_path}")

    def load_labels(self):
        img_path = self.images[self.current_idx]
        label_path = self.output_folder / f"{img_path.stem}.txt"

        self.current_boxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    values = list(map(float, line.strip().split()))
                    if len(values) == 5:
                        self.current_boxes.append(values)
            print(f"Loaded {len(self.current_boxes)} existing boxes")

    def next_image(self):
        self.save_labels()
        if self.current_idx < len(self.images) - 1:
            self.current_idx += 1
            self.load_image()
        else:
            print("Last image reached!")

    def prev_image(self):
        self.save_labels()
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_image()

    def load_image(self):
        img_path = self.images[self.current_idx]
        self.img = cv2.imread(str(img_path))
        if self.img is None:
            raise ValueError(f"Could not load image: {img_path}")

        self.load_labels()
        self.update_display()
        print(f"\nLoaded: {img_path.name}")

    def run(self):
        cv2.namedWindow('YOLO Labeler')
        cv2.setMouseCallback('YOLO Labeler', self.mouse_callback)

        self.load_image()

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # q or ESC
                self.save_labels()
                break
            elif key == ord('n') or key == 32:  # n or Space
                self.next_image()
            elif key == ord('p'):
                self.prev_image()
            elif key == ord('z'):  # Undo last box
                if self.current_boxes:
                    removed = self.current_boxes.pop()
                    print(f"Removed box: {self.classes[int(removed[0])]}")
                    self.update_display()
            elif ord('0') <= key <= ord('9'):  # Number keys for class selection
                class_num = key - ord('0')
                if class_num < len(self.classes):
                    self.current_class = class_num
                    print(f"Switched to class [{self.current_class}]: {self.classes[self.current_class]}")
                    self.update_display()

        cv2.destroyAllWindows()
        print("\n=== Labeling complete! ===")
        print(f"Labels saved to: {self.output_folder}")
        print(f"Classes file: {self.classes_file}")


if __name__ == "__main__":
    # Usage example
    image_folder = "images"  # Change this to your image folder

    if not os.path.exists(image_folder):
        print(f"Creating example folder: {image_folder}")
        print(f"Please add your images to '{image_folder}' and run again")
        os.makedirs(image_folder, exist_ok=True)
    else:
        labeler = YOLOLabeler(image_folder, output_folder="labels", classes_file="classes.txt")
        labeler.run()
