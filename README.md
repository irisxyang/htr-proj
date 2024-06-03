TABLE OF CONTENTS:

- example_images: a couple of example notes, can be submitted to the model for extraction
- static: holds script.js, style.css (for web interface style)
- templates: holds html files for web interface
- app.yaml: required for web app to run
- requirements.txt: required for web app to run
- main.py: web app interface (using Flask)

- image_processing.py: contains all the image processing functions, such as cropping,
  increasing contrast, and detecting ink clusters
- text_extraction.py: contains the full text and sketch extraction pipeline, including all
  the handwriting recognition and bounding box processing functions

INSTRUCTIONS:
To run the web app, go to the root directory and run "python main.py" from terminal
You will need to replace the CREDENTIAL_PATH in text_extraction.py with the path to
the credentials in the zip file (htr-422302-227236128117.json) in order to use the
Google Cloud Vision API
Feel free to use any of the example images in the example_images directory to test
out the interface!
