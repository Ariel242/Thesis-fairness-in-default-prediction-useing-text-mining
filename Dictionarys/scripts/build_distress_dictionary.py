from pathlib import Path

import pandas as pd

DICT_DIR = Path(__file__).resolve().parent.parent / "dictionaries"

# מילון מצוקה/דחיפות ל-LendingClub: מילון LM (Loughran-McDonald) נבנה על דוחות
# 10-K תאגידיים, ואינו מכסה שפת מצוקה אישית (medical, emergency, hardship...).
# הרשימה הבאה נבנתה ידנית מתוך vocabulary_frequencies_lending_club.csv,
# בגישה מחמירה: כל מילה רב-משמעית או נפוצה מדי בהקשרים לא-קשורים (hard,
# trouble, problem, worry, poor, tight, penalty...) הושארה מחוץ לרשימה.
# רק ברמת יוניגרם (מילה בודדת), תואם למבנה LM_Sentiment_Dictionary_Slim.csv.
DISTRESS_WORDS = {
    # רפואי
    "medical", "surgery", "illness", "ill", "sick", "disability", "disable", "cancer",
    # תעסוקה/כלכלה
    "unemployed", "unemployment", "layoff", "recession", "downturn",
    # משפטי/פיננסי-חמור
    "bankruptcy", "foreclosure", "foreclose", "delinquent", "delinquency",
    "overdue", "unpaid", "lien", "collection", "settlement",
    # אירועי חיים
    "divorce", "death", "funeral", "accident",
    # דחיפות/פתאומיות
    "emergency", "unexpected", "unexpectedly", "unforeseen", "unforseen", "asap",
    # מצוקה רגשית/כלכלית מפורשת
    "hardship", "crisis", "struggle", "desperate", "desperately", "stressful",
    "suffer", "drown", "overwhelm", "trap", "crunch", "strain", "burden",
    "destroy", "ruin", "deplete", "juggle", "barely",
}

# טעינת אוצר המילים שחולץ מעמודת ה-DESC הגולמית
df = pd.read_csv(DICT_DIR / "vocabulary_frequencies_lending_club.csv")

df["Distress"] = 0
is_unigram = df["Type"] == "Unigram"
df.loc[is_unigram & df["Term"].isin(DISTRESS_WORDS), "Distress"] = 1

df.to_csv(DICT_DIR / "LC_Distress_Dictionary.csv", index=False)

n_flagged = int(df["Distress"].sum())
print(f"Distress terms flagged: {n_flagged} / {len(DISTRESS_WORDS)} defined")
print(f"Output file: LC_Distress_Dictionary.csv")
