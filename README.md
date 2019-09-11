# CVsudoku

Solves sudoku from an image. Uses `opencv` to process the image and extract digits from the image.
Digits are recognized with a simple CNN implemented in `keras` and trained on custom dataset composed of digits extracted from various images of sudoku boards. Sudoku is solved with Peter Norvig's algorithm described here: https://norvig.com/sudoku.html.

![example](https://i.imgur.com/1oJZ3HN.png)

TODO:
- [x] Extract the sudoku board from the image
- [x] Remove grid lines from the image
- [x] Localize digits 
- [x] Classify digits
- [x] Solve sudoku
- [ ] Display solution as an image
- [ ] Improve image processing
- [ ] Improve digits classification
- [ ] Add 'exe' file
- [ ] Write README
- [x] Add LICENCE
