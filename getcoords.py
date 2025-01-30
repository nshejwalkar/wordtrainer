import cv2

def get_roi_coordinates_from_video(video_path):
   # Load the video
   video = cv2.VideoCapture(video_path)

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
         roi_coordinates.append((x, y))

   # Display the first frame and set the mouse callback
   cv2.imshow("Select ROI", frame)
   cv2.setMouseCallback("Select ROI", select_roi)

   # Wait until the user presses a key, then close
   print("Click on the corners of your ROI, then press any key to finish.")
   cv2.waitKey(0)
   cv2.destroyAllWindows()

   # Release the video
   video.release()

   return roi_coordinates

# Example usage
video_path = "./backend/uploads/iphone12.mp4"
coordinates = get_roi_coordinates_from_video(video_path)
print("Selected coordinates:", coordinates)
