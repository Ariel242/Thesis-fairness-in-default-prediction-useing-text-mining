from pathlib import Path

import pandas as pd

DICT_DIR = Path(__file__).resolve().parent.parent / "dictionaries"

# רשימת ביקורת (לא לניקוד) - כל המילים ש"עלו לדיון" בבניית מילון המצוקה:
# 1) DISTRESS  - 51 המילים שסומנו בפועל ב-LC_Distress_Dictionary.csv
# 2) BORDERLINE - מילים שנפסלו למרות שיש בהן היגיון, כי הן רב-משמעיות מדי
#    או נפוצות מדי בהקשרים לא-קשורים (כולל "default", שנפסל מסיבה שונה -
#    חשש לדליפת מידע מול משתנה המטרה, לא בגלל דו-משמעות).
# לכל מילה יש נימוק קצר להחלטה, כדי שאפשר יהיה לעבור עליה ולהחליט אחרת.

DISTRESS_WORDS = {
    "medical", "surgery", "illness", "ill", "sick", "disability", "disable", "cancer",
    "unemployed", "unemployment", "layoff", "recession", "downturn",
    "bankruptcy", "foreclosure", "foreclose", "delinquent", "delinquency",
    "overdue", "unpaid", "lien", "collection", "settlement",
    "divorce", "death", "funeral", "accident",
    "emergency", "unexpected", "unexpectedly", "unforeseen", "unforseen", "asap",
    "hardship", "crisis", "struggle", "desperate", "desperately", "stressful",
    "suffer", "drown", "overwhelm", "trap", "crunch", "strain", "burden",
    "destroy", "ruin", "deplete", "juggle", "barely",
}

BORDERLINE_WORDS = {
    "relief": "לרוב 'debt/financial relief' אבל גם תחושת הקלה כללית",
    "tight": "לרוב 'תקציב הדוק' אבל יכול לתאר כל דבר הדוק",
    "poor": "poor credit / poor health / עני במילולי - אבל גם 'בחירה גרועה'",
    "loss": "אובדן עבודה/כספי, אבל גם ירידה במשקל, הפסד השקעה וכו'",
    "trouble": "לרוב 'צרות כלכליות' אבל מילה כללית מאוד",
    "hurt": "'hurt credit' נפוץ, אבל יכול להיות פגיעה פיזית",
    "penalty": "קנס על תשלום מאוחר, אבל גם קנס משפטי/ספורט",
    "crash": "תאונת רכב (מצוקה) מול קריסת שוק מול שימוש כללי",
    "pressure": "לחץ כלכלי מול לחץ כללי (עבודה וכו')",
    "battle": "'battle debt' ניב אפשרי, אבל שימוש כללי מדי",
    "fear": "רגש כללי, לא בהכרח פיננסי",
    "worry": "נפוץ מאוד ומאוד כללי - לא מבדיל מצוקה",
    "hole": "'debt hole' ניב אפשרי מול חור פיזי (הקשר שיפוצים חזק בקורפוס)",
    "dig": "'dig out of debt' מול חפירה פיזית (בריכות)",
    "mess": "'בלגן כלכלי' מול בלגן פיזי",
    "hit": "'hit hard times' אפשרי אבל שימוש כללי מדי",
    "knock": "'knock out debt' - זה בעצם חיובי/יזום, לא מצוקה",
    "lack": "'lack of funds' סביר אבל כללי מדי לבד",
    "outrageous": "תלונה על ריבית/עמלה, לא אירוע מצוקה אישי",
    "ridiculous": "תלונה על הלוואה/ריבית, לא על מצב אישי",
    "ridiculously": "כנ״ל",
    "horrible": "שם תואר שלילי כללי מדי",
    "impossible": "כללי, שימושים רבים",
    "negative": "כללי (יתרה שלילית, ציון שלילי וכו')",
    "delay": "עיכוב תשלום אפשרי אבל כללי מדי",
    "drain": "'ניקוז חסכונות' מול ניקוז פיזי (הרבה אוצר מילים של שיפוצים)",
    "decline": "ירידה בהכנסה מול דחיית הצעה וכו'",
    "claim": "תביעת ביטוח - נייטרלי/כללי",
    "damage": "פגיעה באשראי מול נזק פיזי לבית (הקשר שיפוצים)",
    "deny": "נדחה על משהו - אשראי, תביעה, בקשה - כללי מדי",
    "chapter": "פרק בפשיטת רגל (Chapter 7/13) מול פרק בספר/בחיים",
    "restructure": "יכול להיות מחזור הלוואה יזום, לא בהכרח מצוקה",
    "wipe": "'wipe out debt' זה בעצם חיובי - סילוק חוב",
    "erase": "כנ״ל, קונוטציה חיובית (סילוק חוב)",
    "stave": "'stave off' נדיר ולא ברור מספיק",
    "quit": "עזיבת עבודה יכולה להיות מרצון, לא מצוקה",
    "downsize": "צמצום משרה (מצוקה) מול צמצום בית (בחירה נייטרלית)",
    "hurricane": "נזק אסון (מצוקה) מול מימון הגנה מפני סערות (יזום)",
    "flood": "כנ״ל",
    "storm": "כנ״ל, וגם שימוש כללי מדי",
    "die": "כללי מדי כמילה בודדת",
    "fell": "כללי מדי (נפילה, לא בהכרח 'נפילה כלכלית')",
    "felt": "כללי מדי",
    "veteran": "תיאור סטטוס/תעסוקה, לא מצוקה בעצמו",
    "hospital": "לרוב תיאור מקום עבודה ('אני עובד בבית חולים'), לא צורך רפואי",
    "doctor": "לרוב תיאור מקצוע ('אני רופא'), לא הוצאה רפואית",
    "dentist": "כנ״ל",
    "nurse": "כנ״ל - מופיע הרבה בהקשר תעסוקתי בקורפוס",
    "procedure": "יכול להיות 'הליך רפואי' אבל גם 'הליך הלוואה'",
    "payday": "מציין סוג הלוואה (payday loan), לא מתאר את מצב הלווה",
    "gap": "פער תעסוקתי/אשראי/פיזי - כללי מדי",
    "upside": "'upside down' על הלוואה, אבל המילה הבודדת מעורפלת מדי",
    "sink": "כיור (הקשר שיפוצים) מול 'שוקע בחובות'",
    "weight": "'משקל החוב' ניב מול משקל פיזי",
    "shoulder": "'לשאת בנטל' ניב מול איבר גוף",
    "stretch": "'למתוח תקציב' ניב מול פועל כללי",
    "grind": "בעיקר 'משאבת גריינדר' (אינסטלציה) בקורפוס הזה, לא 'שגרה מתישה'",
    "worst": "סופרלטיב כללי",
    "worse": "כנ״ל",
    "difficult": "'difficult time' כביגרם מרמז, אבל המילה הבודדת כללית מדי",
    "tough": "כללי (החלטה קשה מול תקופה קשה)",
    "fail": "פועל כללי מדי",
    "error": "כללי מדי",
    "mistake": "כללי מדי ('make mistake')",
    "prevent": "פועל תכנון יזום, לא סמן מצוקה",
    "avoid": "כנ״ל - תכנון יזום (הימנעות מריבית גבוהה)",
    "regret": "רגש כללי",
    "coverage": "מונח ביטוח כללי",
    "bite": "לרוב 'little bite' = 'קצת', לא קשור למצוקה",
    "catch": "'catch up on payments' אפשרי אבל כללי מדי לבד",
    "concern": "כללי מדי",
    "issue": "כללי מאוד (payment issue כביגרם ספציפי יותר)",
    "hard": "נפוץ וכללי ביותר - בעיקר 'קשה' או 'עובד קשה', לא מצוקה",
    "problem": "כללי ביותר - יכול להתייחס לכל דבר",
    "default": "לא דו-משמעות סמנטית אלא חשש לדליפת מידע - המילה חופפת למשתנה המטרה של התזה (חיזוי default)",
}

freq = (
    pd.read_csv(DICT_DIR / "vocabulary_frequencies_lending_club.csv")
    .query("Type == 'Unigram'")
    .set_index("Term")["Document_Frequency"]
)

rows = []
for w in sorted(DISTRESS_WORDS):
    rows.append({
        "Term": w,
        "Document_Frequency": int(freq.get(w, 0)),
        "Category": "Distress",
        "Reason": "נכלל במילון המצוקה הסופי (LC_Distress_Dictionary.csv)",
    })
for w, reason in sorted(BORDERLINE_WORDS.items()):
    rows.append({
        "Term": w,
        "Document_Frequency": int(freq.get(w, 0)),
        "Category": "Borderline_Excluded",
        "Reason": reason,
    })

review = pd.DataFrame(rows).sort_values(
    by=["Category", "Document_Frequency"], ascending=[True, False]
)
review.to_csv(DICT_DIR / "LC_Distress_Words_Review.csv", index=False, encoding="utf-8-sig")

print(f"Distress:             {(review['Category'] == 'Distress').sum()}")
print(f"Borderline_Excluded:  {(review['Category'] == 'Borderline_Excluded').sum()}")
print("Output file: LC_Distress_Words_Review.csv")
