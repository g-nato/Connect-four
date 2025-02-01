# Connect Four ML Prediction Model

## Overview
This project aims to predict the outcome of a **Connect Four** game using machine learning. The game is played on a 6x7 grid, where two players (red and yellow) aim to align four consecutive pieces either horizontally, vertically, or diagonally. The task is to classify the outcome based on the game state after the first eight moves, with possible outcomes being **Red Win (R)**, **Yellow Win (Y)**, or **Draw (.)**.

## Dataset
The dataset contains game states represented as strings, each encoding a 6x7 grid. The grid positions can contain:
- `R` for Red disc
- `Y` for Yellow disc
- `.` for Empty space

Each string also includes the outcome of the game as the last character: `R` for a red win, `Y` for a yellow win, and `.` for a draw.

### Example Configuration:
A typical game state might look like this:
```
YY.... Y..... RYRR.. ...... ...... R..... R
```

In this example:
- The grid is represented by the first 42 characters (6 rows x 7 columns).
- The last character indicates the outcome of the game: in this case, a **Red win (R)**.

### Distribution of Outcomes:
- **Training**: Red Win (43,141), Yellow Win (16,173), Draw (6,243)
- **Test**: Red Win (665), Yellow Win (229), Draw (106)
- **Validation**: Red Win (667), Yellow Win (233), Draw (100)

