# CVsudoku

Solves sudoku from an image. Uses `opencv` to process the image and extract digits from the image.
Digits are recognized with a simple CNN implemented in `keras` and trained on custom dataset composed of digits extracted from various images of sudoku boards. Sudoku is solved with Peter Norvig's algorithm described here: https://norvig.com/sudoku.html.

![example](https://i.imgur.com/1oJZ3HN.png)

## Usage
Clone the repo: `git clone https://github.com/wieczszy/CVsudoku.git`

Install requirements: `pip install -r requirements.txt`

Run the solver on a selected image: `python app.py example_sudoku.png`

An H5 file with weights for the digit recognition model is provided and used as default. If you wan to use other file with weights you can point to it using a `-w` flag. For example: `python app.py example_sudoku.png -w my_model.h5`

Remember that the weights have to be compatible with the CNN model:

```python
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
```

## Accuracy
Right now it works well with good quality sudoku images but sometimes struggles without a clear reason. Not really usefull for low quality IRL images. Image processing and digit recognition are being worked on - maybe it will be improved maybe not. 

## Notes to self
TODO:
- [x] Extract the sudoku board from the image
- [x] Remove grid lines from the image
- [x] Localize digits 
- [x] Classify digits
- [x] Solve sudoku
- [ ] Display solution as an image
- [ ] Improve image processing
- [ ] Improve digits classification
- [x] Add 'exe' file
- [ ] Write README
- [x] Add LICENCE
- [ ] Exception handling for solving/image reading errors
- [ ] Describe/publish dataset
