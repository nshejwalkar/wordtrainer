import cv2
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from opencv_processing import _apply_preprocessing_board, _magic_to_pixels_image, _get_roi

def get_roi_coordinates_from_video(video_path):
   # Load the video
   video = cv2.VideoCapture(video_path)
   width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
   height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  
   # max_display_width = 400

   # window_name = 'Frame'
   # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Make window resizable

   # X IS WIDTH, Y IS HEIGHT
   print(f"Video dimensions: {width}x{height}")

   # Check if the video loaded successfully
   if not video.isOpened():
      print("Error: Could not open video.")
      return []

   # Read the first frame from the video
   for _ in range(200):
      _, frame = video.read()

   # Create a list to store the coordinates
   roi_coordinates = []

   def select_roi(event, x, y, flags, param):
      if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
         print(f"Clicked at: ({x}, {y})")
         print(f"Relative ratio: {x/width}, Height: {y/height}")
         roi_coordinates.append((x, y))  # Store the coordinates as ratios of width and height

   # # Resize frame if it's too large
   # height, width = frame.shape[:2]
   # # if width > max_display_width:
   # scale = max_display_width / width
   # frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

   # Display the first frame and set the mouse callback
   cv2.imshow("Select ROI", frame)
   cv2.setMouseCallback("Select ROI", select_roi)

   # Wait until the user presses a key, then close
   print("Click on the corners of your ROI, then press any key to finish.")
   cv2.waitKey(0)
   cv2.destroyAllWindows()

   xleft, ytop = min(coord[0] for coord in roi_coordinates), min(coord[1] for coord in roi_coordinates)
   xright, ybottom = max(coord[0] for coord in roi_coordinates), max(coord[1] for coord in roi_coordinates)
   roi_coordinates = [(xleft, ytop), (xright, ytop), (xright, ybottom), (xleft, ybottom)]
   print(f"Selected ROI coordinates: {roi_coordinates}")
   print(f"ytop: {ytop/height}\nybottom: {ybottom/height}\nxleft: {xleft/width}\nxright: {xright/width}")
   print(f"number of pixels: {(xright - xleft)*(ybottom - ytop)}")

   # Release the video
   video.release()

   return roi_coordinates

def get_roi_coordinates_from_image(image_path):
   # Load the video
   image = cv2.imread(image_path)
   height, width = image.shape[:2]
   magic_pix = _magic_to_pixels_image(image_path)
   board_roi = _get_roi(image, magic_pix, 'board')
   board_roi = _apply_preprocessing_board(board_roi, modality='image', shape=image.shape)
   # max_display_width = 400

   # window_name = 'Frame'
   # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Make window resizable

   # X IS WIDTH, Y IS HEIGHT
   print(f"image dimensions: {width}x{height}")

   # Create a list to store the coordinates
   roi_coordinates = []

   def select_roi(event, x, y, flags, param):
      if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
         print(f"Clicked at: ({x}, {y})")
         print(f"Relative ratio: {x/width}, Height: {y/height}")
         roi_coordinates.append((x, y))  # Store the coordinates as ratios of width and height

   cv2.imshow("Select ROI", board_roi)
   cv2.setMouseCallback("Select ROI", select_roi)

   # Wait until the user presses a key, then close
   print("Click on the corners of your ROI, then press any key to finish.")
   cv2.waitKey(0)
   cv2.destroyAllWindows()

   xleft, ytop = min(coord[0] for coord in roi_coordinates), min(coord[1] for coord in roi_coordinates)
   xright, ybottom = max(coord[0] for coord in roi_coordinates), max(coord[1] for coord in roi_coordinates)
   roi_coordinates = [(xleft, ytop), (xright, ytop), (xright, ybottom), (xleft, ybottom)]
   print(f"Selected ROI coordinates: {roi_coordinates}")
   print(f"ytop: {ytop/height}\nybottom: {ybottom/height}\nxleft: {xleft/width}\nxright: {xright/width}")
   print(f"number of pixels: {(xright - xleft)*(ybottom - ytop)}")

   return roi_coordinates

# Example usage
video_path = "../uploads/iphone12.mp4"
image_path = "../../models_data/pics/IMG_6358.PNG"
# coordinates = get_roi_coordinates_from_video(video_path)

get_roi_coordinates_from_image(image_path)