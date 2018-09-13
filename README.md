## Captcha Solver
 Train and solve captchas of UP Scholarship website using tensorflow.

 > Created for educational purpose. Please don't use it to spam the website.

### Requirements
* _Python >= 3_
* _tensorflow_
* _requests_
* _opencv_python_
* _numpy_
* _imutils_
* _Pillow_
* _scikit_learn_
 
The python libraries are listed in requirements.txt

> Captcha images are saved in 'saved_captchas' folder for training.

### Steps
1. Run `python transform.py` to clean images and save them in '_transformed_captchas_' folder.
2. Run `python extract_single_letters_from_captchas.py` to extract individual characters and put them '_extracted_letter_images_' folder. Letter images are saved inside letter named folder.
3. Run `python train_captcha_model.py` to train the model.
4. Run `python solve_captchas_using_model.py` to test captcha by downloading it from the website.