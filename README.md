# Interactive Macronutrient Predictor

An interactive Streamlit dashboard for exploring personal nutrition and lifestyle data — built as a **Data Visualization Techniques** course project.

The app combines several interactive visualizations with a lightweight **K-Nearest Neighbors (K-NN)** model that matches a day of macronutrient intake to the person whose eating pattern it most resembles.

---

## Features

| Module | Description |
|--------|-------------|
| **Stacked Bar Chart** | Daily calorie structure broken down by meal (breakfast → dinner) and macronutrients (protein / fat / carbs). Supports person filtering and date-range brushing. |
| **Radar Chart** | Macronutrient balance (protein, fat, carbs, sugars, fiber, salt) for one or more people over a selected period. Values are scaled relative to the dataset maximum. |
| **Productivity Heatmap** | Calendar-style heatmap scoring each day (0–100) based on how well diet targets, study hours, and step goals were met. |
| **Scatter Plot** | Explore correlations between any two metrics (calories, macros, steps, study hours). |
| **ML Predictor (K-NN)** | Enter a day's macros via sliders; the model finds the 5 most similar historical days (Euclidean distance in 6-D space after Min-Max scaling) and predicts whose eating pattern it matches. |

---

## Dataset

Three participants tracked their daily meals and lifestyle metrics:

- **Hubert**, **Szymon**, **Zosia**

Each CSV row represents a single meal with the following columns:

| Column | Description |
|--------|-------------|
| `data` | Date (`DD.MM.YYYY`) |
| `pora_dnia` | Meal slot (1–5: breakfast, second breakfast, lunch, afternoon snack, dinner) |
| `kalorie` | Calories (kcal) |
| `bialko` | Protein (g) |
| `tluszcze` | Fat (g) |
| `weglowodany` | Carbohydrates (g) |
| `cukry` | Sugars (g) |
| `blonnik` | Fiber (g) |
| `sol` | Salt (g) |
| `godz_na_ucz` | Study hours that day |
| `ilosc_krokow` | Step count that day |

Files use `;` as the separator and `,` as the decimal mark (Polish locale).

---

## Tech Stack

- **[Streamlit](https://streamlit.io/)** — interactive web UI
- **[Plotly](https://plotly.com/python/)** — charts (bar, radar, heatmap, scatter)
- **[Pandas](https://pandas.pydata.org/)** / **[NumPy](https://numpy.org/)** — data wrangling and the from-scratch K-NN implementation

---

## Project Structure

```
Interactive-Macronutrient-Predictor/
├── app.py                          # Main Streamlit entry point
├── modules/
│   ├── stacked_bar_chart.py        # Meal/calorie stacked bars
│   ├── radar_chart.py              # Macronutrient radar
│   ├── heatmap.py                  # Productivity calendar heatmap
│   ├── scatter_plot.py             # Correlation scatter
│   └── prediction.py               # K-NN personality predictor
├── avatars/                        # Profile images for the ML module
├── dane_hubert.csv                 # Hubert's tracking data
├── dane_szymon.csv                 # Szymon's tracking data
├── dane_zosia.csv                  # Zosia's tracking data
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+

### Installation

```bash
git clone https://github.com/domanskis06/Interactive-Macronutrient-Predictor.git
cd Interactive-Macronutrient-Predictor

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Run the app

```bash
streamlit run app.py
```

The dashboard opens in your browser at `http://localhost:8501`.

---

## How the K-NN Predictor Works

1. Meals are aggregated to **daily totals** per person.
2. Incomplete / outlier days are filtered out.
3. Features (`protein`, `fat`, `carbs`, `sugars`, `fiber`, `salt`) are **Min-Max scaled** to `[0, 1]`.
4. For a user-provided day vector, the model computes **Euclidean distance** to every historical day and selects the **k = 5** nearest neighbors.
5. A **distance-weighted vote** picks the most similar person; similarity % is derived from the closest match.

---

## Authors

Course project for **Techniki Wizualizacji Danych (TWD)** — semester 3.

Contributors: Hubert, Szymon, Zosia.

---

## License

This project is shared for educational and portfolio purposes.
