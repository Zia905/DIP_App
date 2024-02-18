import streamlit as st
from PIL import Image, ImageFilter
import cv2
import numpy as np

# Set page title and icon
st.set_page_config(page_title='Image Processing App', page_icon=':camera:')

# Page title and description
st.title('Welcome to Image Processor!')
st.write("Upload the Image & click on the buttons below to apply different image processing techniques.")

# Sidebar title
st.sidebar.title('Upload Image')

# Upload image in the sidebar
uploaded_image = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Main title
st.write('Image Processing Options')

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.sidebar.image(image, caption="Uploaded Image", use_column_width=True)

    # Image processing options
    # First row of buttons
    col1, col2, col3 = st.columns(3)
    if col1.button('Grayscale'):
        # Convert the image to grayscale
        grayscale_image = image.convert('L')
        st.image(grayscale_image, caption="Grayscale Image", use_column_width=True)

    if col2.button('Blur'):
        # Apply a blur filter to the image
        blurred_image = image.filter(ImageFilter.BLUR)
        st.image(blurred_image, caption="Blurred Image", use_column_width=True)

    if col3.button('Flip'):
        # Flip the image horizontally
        flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        st.image(flipped_image, caption="Flipped Image", use_column_width=True)

    # Second row of buttons
    col4, col5, col6 = st.columns(3)
    if col4.button('Face Detection'):
        # Convert the image to OpenCV format and perform face detection
        cv_image = np.array(image)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

        # Load the pre-trained face detection model
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(cv_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Convert the image back to PIL format
        face_detected_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        st.image(face_detected_image, caption="Face Detection", use_column_width=True)

    if col5.button('Edge Detection'):
        # Convert the image to OpenCV format
        cv_image = np.array(image)
        cv_image_gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(cv_image_gray, 100, 200)

        # Convert the edges image back to PIL format
        edge_detected_image = Image.fromarray(edges)
        st.image(edge_detected_image, caption="Edge Detection", use_column_width=True)

    if col6.button('Noise Reduction'):
        # Apply noise reduction using a bilateral filter
        noise_reduced_image = cv2.bilateralFilter(np.array(image), 9, 75, 75)
        st.image(noise_reduced_image, caption="Noise Reduced Image", use_column_width=True)

    # Third row of buttons
    col7, col8, col9 = st.columns(3)
    if col7.button('Color Enhancement'):
        # Enhance the color of the image
        enhanced_image = cv2.convertScaleAbs(np.array(image), alpha=2.0, beta=0)
        st.image(enhanced_image, caption="Color Enhanced Image", use_column_width=True)

    if col8.button('Image Rotation'):
        # Rotate the image by 90 degrees
        rotated_image = image.rotate(90)
        st.image(rotated_image, caption="Rotated Image", use_column_width=True)

    if col9.button('Image Cropping'):
        # Crop the image to a square
        width, height = image.size
        size = min(width, height)
        cropped_image = image.crop(
            ((width - size) // 2, (height - size) // 2, (width + size) // 2, (height + size) // 2))
        st.image(cropped_image, caption="Cropped Image", use_column_width=True)
