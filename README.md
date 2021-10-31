"" image detection and crop Project""

To install:
1. download mini conda via link: https://docs.conda.io/en/latest/miniconda.html
2. install conda via installer
3. open terminal or cmd
4.. navigate to directory of the project eg: " cd C:*path to directory*\image_detection"
change path to directory according to yr directory where the project is
5. run command: " conda create -n my_env python =3.9 "
press y if any prompt is recieved wait for packages to install
6. run command:  " conda activate my_env "
""""" then follow operate instructions

To operate:
1. Install requirments via command  "pip3 install -r requirements. txt"
2. run app.py via ccommand "python app.py "
3. run index.html and upload file
4. Model take the files and detect images and save them to the directory cropped with names like  filename(no of image).jpg
example if 1.jpg has 2 images it is saved as 1(1).jpg and 1(2).jpg
5. It returns a JSON response containing the list of all the paths of extracted images
