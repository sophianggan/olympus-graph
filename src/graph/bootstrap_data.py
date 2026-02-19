"""
Olympus Graph – Sample Data Bootstrap

Creates a small `athlete_events.csv` in `data/raw/` so the full project
can run end-to-end without downloading Kaggle data first.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from src.config import DATA_RAW_DIR


SAMPLE_ROWS = [
    # 2012 London
    {"ID": 1, "Name": "Usain Bolt", "Sex": "M", "Age": 25, "Height": 195, "Weight": 94, "Team": "Jamaica", "NOC": "JAM", "Games": "2012 Summer", "Year": 2012, "Season": "Summer", "City": "London", "Sport": "Athletics", "Event": "Athletics Men's 100 metres", "Medal": "Gold"},
    {"ID": 2, "Name": "Yohan Blake", "Sex": "M", "Age": 22, "Height": 180, "Weight": 73, "Team": "Jamaica", "NOC": "JAM", "Games": "2012 Summer", "Year": 2012, "Season": "Summer", "City": "London", "Sport": "Athletics", "Event": "Athletics Men's 100 metres", "Medal": "Silver"},
    {"ID": 3, "Name": "Justin Gatlin", "Sex": "M", "Age": 30, "Height": 185, "Weight": 83, "Team": "United States", "NOC": "USA", "Games": "2012 Summer", "Year": 2012, "Season": "Summer", "City": "London", "Sport": "Athletics", "Event": "Athletics Men's 100 metres", "Medal": "Bronze"},
    {"ID": 4, "Name": "Shelly-Ann Fraser-Pryce", "Sex": "F", "Age": 25, "Height": 152, "Weight": 52, "Team": "Jamaica", "NOC": "JAM", "Games": "2012 Summer", "Year": 2012, "Season": "Summer", "City": "London", "Sport": "Athletics", "Event": "Athletics Women's 100 metres", "Medal": "Gold"},
    {"ID": 5, "Name": "Carmelita Jeter", "Sex": "F", "Age": 32, "Height": 163, "Weight": 61, "Team": "United States", "NOC": "USA", "Games": "2012 Summer", "Year": 2012, "Season": "Summer", "City": "London", "Sport": "Athletics", "Event": "Athletics Women's 100 metres", "Medal": "Silver"},
    {"ID": 6, "Name": "Veronica Campbell-Brown", "Sex": "F", "Age": 30, "Height": 163, "Weight": 58, "Team": "Jamaica", "NOC": "JAM", "Games": "2012 Summer", "Year": 2012, "Season": "Summer", "City": "London", "Sport": "Athletics", "Event": "Athletics Women's 100 metres", "Medal": "Bronze"},
    # 2016 Rio
    {"ID": 1, "Name": "Usain Bolt", "Sex": "M", "Age": 29, "Height": 195, "Weight": 94, "Team": "Jamaica", "NOC": "JAM", "Games": "2016 Summer", "Year": 2016, "Season": "Summer", "City": "Rio de Janeiro", "Sport": "Athletics", "Event": "Athletics Men's 100 metres", "Medal": "Gold"},
    {"ID": 2, "Name": "Justin Gatlin", "Sex": "M", "Age": 34, "Height": 185, "Weight": 83, "Team": "United States", "NOC": "USA", "Games": "2016 Summer", "Year": 2016, "Season": "Summer", "City": "Rio de Janeiro", "Sport": "Athletics", "Event": "Athletics Men's 100 metres", "Medal": "Silver"},
    {"ID": 7, "Name": "Andre De Grasse", "Sex": "M", "Age": 21, "Height": 176, "Weight": 70, "Team": "Canada", "NOC": "CAN", "Games": "2016 Summer", "Year": 2016, "Season": "Summer", "City": "Rio de Janeiro", "Sport": "Athletics", "Event": "Athletics Men's 100 metres", "Medal": "Bronze"},
    {"ID": 8, "Name": "Elaine Thompson-Herah", "Sex": "F", "Age": 24, "Height": 167, "Weight": 57, "Team": "Jamaica", "NOC": "JAM", "Games": "2016 Summer", "Year": 2016, "Season": "Summer", "City": "Rio de Janeiro", "Sport": "Athletics", "Event": "Athletics Women's 100 metres", "Medal": "Gold"},
    {"ID": 9, "Name": "Tori Bowie", "Sex": "F", "Age": 25, "Height": 175, "Weight": 61, "Team": "United States", "NOC": "USA", "Games": "2016 Summer", "Year": 2016, "Season": "Summer", "City": "Rio de Janeiro", "Sport": "Athletics", "Event": "Athletics Women's 100 metres", "Medal": "Silver"},
    {"ID": 10, "Name": "Shelly-Ann Fraser-Pryce", "Sex": "F", "Age": 29, "Height": 152, "Weight": 52, "Team": "Jamaica", "NOC": "JAM", "Games": "2016 Summer", "Year": 2016, "Season": "Summer", "City": "Rio de Janeiro", "Sport": "Athletics", "Event": "Athletics Women's 100 metres", "Medal": "Bronze"},
    # 2020 Tokyo
    {"ID": 11, "Name": "Lamont Marcell Jacobs", "Sex": "M", "Age": 26, "Height": 188, "Weight": 80, "Team": "Italy", "NOC": "ITA", "Games": "2020 Summer", "Year": 2020, "Season": "Summer", "City": "Tokyo", "Sport": "Athletics", "Event": "Athletics Men's 100 metres", "Medal": "Gold"},
    {"ID": 12, "Name": "Fred Kerley", "Sex": "M", "Age": 26, "Height": 191, "Weight": 79, "Team": "United States", "NOC": "USA", "Games": "2020 Summer", "Year": 2020, "Season": "Summer", "City": "Tokyo", "Sport": "Athletics", "Event": "Athletics Men's 100 metres", "Medal": "Silver"},
    {"ID": 13, "Name": "Andre De Grasse", "Sex": "M", "Age": 26, "Height": 176, "Weight": 70, "Team": "Canada", "NOC": "CAN", "Games": "2020 Summer", "Year": 2020, "Season": "Summer", "City": "Tokyo", "Sport": "Athletics", "Event": "Athletics Men's 100 metres", "Medal": "Bronze"},
    {"ID": 14, "Name": "Elaine Thompson-Herah", "Sex": "F", "Age": 29, "Height": 167, "Weight": 57, "Team": "Jamaica", "NOC": "JAM", "Games": "2020 Summer", "Year": 2020, "Season": "Summer", "City": "Tokyo", "Sport": "Athletics", "Event": "Athletics Women's 100 metres", "Medal": "Gold"},
    {"ID": 15, "Name": "Shelly-Ann Fraser-Pryce", "Sex": "F", "Age": 34, "Height": 152, "Weight": 52, "Team": "Jamaica", "NOC": "JAM", "Games": "2020 Summer", "Year": 2020, "Season": "Summer", "City": "Tokyo", "Sport": "Athletics", "Event": "Athletics Women's 100 metres", "Medal": "Silver"},
    {"ID": 16, "Name": "Shericka Jackson", "Sex": "F", "Age": 27, "Height": 173, "Weight": 61, "Team": "Jamaica", "NOC": "JAM", "Games": "2020 Summer", "Year": 2020, "Season": "Summer", "City": "Tokyo", "Sport": "Athletics", "Event": "Athletics Women's 100 metres", "Medal": "Bronze"},
    # 2024 Paris (hold-out style rows available for evaluation)
    {"ID": 17, "Name": "Noah Lyles", "Sex": "M", "Age": 27, "Height": 180, "Weight": 70, "Team": "United States", "NOC": "USA", "Games": "2024 Summer", "Year": 2024, "Season": "Summer", "City": "Paris", "Sport": "Athletics", "Event": "Athletics Men's 100 metres", "Medal": "Gold"},
    {"ID": 18, "Name": "Kishane Thompson", "Sex": "M", "Age": 23, "Height": 185, "Weight": 80, "Team": "Jamaica", "NOC": "JAM", "Games": "2024 Summer", "Year": 2024, "Season": "Summer", "City": "Paris", "Sport": "Athletics", "Event": "Athletics Men's 100 metres", "Medal": "Silver"},
    {"ID": 19, "Name": "Fred Kerley", "Sex": "M", "Age": 29, "Height": 191, "Weight": 79, "Team": "United States", "NOC": "USA", "Games": "2024 Summer", "Year": 2024, "Season": "Summer", "City": "Paris", "Sport": "Athletics", "Event": "Athletics Men's 100 metres", "Medal": "Bronze"},
    {"ID": 20, "Name": "Julien Alfred", "Sex": "F", "Age": 23, "Height": 170, "Weight": 58, "Team": "Saint Lucia", "NOC": "LCA", "Games": "2024 Summer", "Year": 2024, "Season": "Summer", "City": "Paris", "Sport": "Athletics", "Event": "Athletics Women's 100 metres", "Medal": "Gold"},
    {"ID": 21, "Name": "Sha'Carri Richardson", "Sex": "F", "Age": 24, "Height": 155, "Weight": 50, "Team": "United States", "NOC": "USA", "Games": "2024 Summer", "Year": 2024, "Season": "Summer", "City": "Paris", "Sport": "Athletics", "Event": "Athletics Women's 100 metres", "Medal": "Silver"},
    {"ID": 22, "Name": "Melissa Jefferson", "Sex": "F", "Age": 23, "Height": 160, "Weight": 52, "Team": "United States", "NOC": "USA", "Games": "2024 Summer", "Year": 2024, "Season": "Summer", "City": "Paris", "Sport": "Athletics", "Event": "Athletics Women's 100 metres", "Medal": "Bronze"},
    # Non-medal participants to provide negatives
    {"ID": 23, "Name": "Trayvon Bromell", "Sex": "M", "Age": 21, "Height": 175, "Weight": 72, "Team": "United States", "NOC": "USA", "Games": "2016 Summer", "Year": 2016, "Season": "Summer", "City": "Rio de Janeiro", "Sport": "Athletics", "Event": "Athletics Men's 100 metres", "Medal": None},
    {"ID": 24, "Name": "Akani Simbine", "Sex": "M", "Age": 27, "Height": 175, "Weight": 67, "Team": "South Africa", "NOC": "RSA", "Games": "2020 Summer", "Year": 2020, "Season": "Summer", "City": "Tokyo", "Sport": "Athletics", "Event": "Athletics Men's 100 metres", "Medal": None},
    {"ID": 25, "Name": "Marie-Josee Ta Lou", "Sex": "F", "Age": 33, "Height": 159, "Weight": 57, "Team": "Ivory Coast", "NOC": "CIV", "Games": "2020 Summer", "Year": 2020, "Season": "Summer", "City": "Tokyo", "Sport": "Athletics", "Event": "Athletics Women's 100 metres", "Medal": None},
    {"ID": 26, "Name": "Dina Asher-Smith", "Sex": "F", "Age": 29, "Height": 164, "Weight": 56, "Team": "Great Britain", "NOC": "GBR", "Games": "2024 Summer", "Year": 2024, "Season": "Summer", "City": "Paris", "Sport": "Athletics", "Event": "Athletics Women's 100 metres", "Medal": None},
]


def create_sample_athlete_events_csv() -> str:
    """Write a minimal athlete_events.csv and return its path."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_RAW_DIR / "athlete_events.csv"
    df = pd.DataFrame(SAMPLE_ROWS)
    df.to_csv(output_path, index=False)
    logger.success(f"Wrote sample dataset: {output_path} ({len(df)} rows)")
    return str(output_path)


if __name__ == "__main__":
    create_sample_athlete_events_csv()
