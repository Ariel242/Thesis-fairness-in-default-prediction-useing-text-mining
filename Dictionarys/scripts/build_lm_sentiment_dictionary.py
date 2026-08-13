from pathlib import Path

import pandas as pd

DICT_DIR = Path(__file__).resolve().parent.parent / "dictionaries"

# טעינת הקובץ המלא (יש להוריד אותו מחדש מ-https://sraf.nd.edu/loughranmcdonald-master-dictionary/
# ולהניח אותו תחת Dictionarys/dictionaries/ אם ברצונך לבנות מחדש את הקובץ הרזה)
df = pd.read_csv(DICT_DIR / "Loughran-McDonald_MasterDictionary_1993-2025.csv")

# בחירת העמודות הרלוונטיות בלבד: מזהה, מילה, ועמודות הסנטימנט
sentiment_columns = [
    "Seq_num",
    "Word",
    "Negative",
    "Positive",
    "Uncertainty",
    "Litigious",
    "Strong_Modal",
    "Weak_Modal",
    "Constraining",
]

# יצירת DataFrame מצומצם
df_slim = df[sentiment_columns]

# שמירה לקובץ רזה חדש שתוכל לעבוד איתו תמיד
df_slim.to_csv(DICT_DIR / "LM_Sentiment_Dictionary_Slim.csv", index=False)
