import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# ASL alphabet (excluding J and Z which require motion)
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
dataset_size = 200  # 200 images per letter
image_size = (224, 224)  # Standard square size for ML models

cap = cv2.VideoCapture(0)  # Using default camera (0)

try:
    for letter in letters:
        # Create directory for current letter
        letter_dir = os.path.join(DATA_DIR, letter)
        if not os.path.exists(letter_dir):
            os.makedirs(letter_dir)

        print(f'Collecting data for letter {letter}')
        print('Press "Q" to quit at any time')
        
        # Wait for user to be ready
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Resize frame for display (keeping aspect ratio)
            height, width = frame.shape[:2]
            display_width = 640
            display_height = int(display_width * (height/width))
            display_frame = cv2.resize(frame, (display_width, display_height))
                
            cv2.putText(display_frame, f'Ready for letter {letter}? Press "Q" to start!', (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(display_frame, 'Press "ESC" to quit', (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.imshow('frame', display_frame)
            key = cv2.waitKey(25)
            if key == ord('q'):
                break
            elif key == 27:  # ESC key
                raise KeyboardInterrupt

        # Collect images for current letter
        counter = 0
        while counter < dataset_size:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Resize frame for display (keeping aspect ratio)
            height, width = frame.shape[:2]
            display_width = 640
            display_height = int(display_width * (height/width))
            display_frame = cv2.resize(frame, (display_width, display_height))
                
            # Display current progress
            cv2.putText(display_frame, f'Letter: {letter} Count: {counter}/{dataset_size}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, 'Press "ESC" to quit', (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
            
            cv2.imshow('frame', display_frame)
            key = cv2.waitKey(25)
            
            if key == 27:  # ESC key
                raise KeyboardInterrupt
            
            # Save the image (square size)
            img_path = os.path.join(letter_dir, f'{counter}.jpg')
            # First resize to maintain aspect ratio
            height, width = frame.shape[:2]
            size = min(height, width)
            start_x = (width - size) // 2
            start_y = (height - size) // 2
            cropped = frame[start_y:start_y+size, start_x:start_x+size]
            # Then resize to final size
            resized_frame = cv2.resize(cropped, image_size)
            cv2.imwrite(img_path, resized_frame)
            print(f'Saved {img_path}')
            
            counter += 1

        print(f'Completed collecting images for letter {letter}')

except KeyboardInterrupt:
    print("\nImage collection stopped by user")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Image collection completed!")
