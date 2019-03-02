from src.detector import detect_faces
from src.utils import show_bboxes
from PIL import Image

def main():
    image = Image.open('images/test3.jpg')
    bounding_boxes, landmarks = detect_faces(image)
    image = show_bboxes(image, bounding_boxes, landmarks)
    image.show()

if __name__ == "__main__":
    main()
