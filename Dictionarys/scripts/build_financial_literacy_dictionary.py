from pathlib import Path

import pandas as pd

DICT_DIR = Path(__file__).resolve().parent.parent / "dictionaries"

# מילון "אוריינות פיננסית" ל-LendingClub: עד כמה הכותב שולט במונחים פיננסיים
# מדויקים/טכניים (APR, DTI, FICO, principal, collateral...) - להבדיל ממילים
# בסיסיות שכולם משתמשים בהן בלי קשר לרמת ידע (rate, credit, payment, score,
# balance, tax, save, budget) ולכן אין להן כוח הבחנה (discriminating power).
#
# הערות מתודולוגיות חשובות:
# 1. "interest" (מאוית נכון) לא מופיע כלל באוצר המילים - כנראה נחסם על ידי
#    max_df=0.70 בסקריפט extract_vocabulary.py (מופיע ביותר מ-70% מהתיאורים).
#    רק הכתיב השגוי "intrest" קיים, ולכן לא נכלל (איות שגוי לא מעיד על אוריינות).
# 2. invest/investment/investor הוצאו בכוונה: ב-LendingClub אלו הביטויים
#    שהלווה פונה בהם ל"משקיעים" (המלווים בפועל בפלטפורמת P2P) - כמו
#    "Dear investors, thank you..." - זה לא מעיד על ידע השקעות של הלווה עצמו.
# 3. מונחים עם ספרות (כמו 401k) לא יכולים להופיע כלל בקובץ המקור כי
#    extract_vocabulary.py משתמש ב-token_pattern של אותיות בלבד.
# 4. רק ברמת יוניגרם, כמו במילוני LM/Distress הקודמים.

LITERACY_WORDS = {
    # מונחי אשראי/הלוואה טכניים
    "apr", "dti", "fico", "heloc", "installment", "principal", "collateral",
    "unsecured", "utilization", "revolve", "lien", "worthiness", "creditor",
    "refinance", "escrow",
    # חשבונאות/הכנסה
    "compound", "liquidity", "gross", "disposable",
    # פרישה/חיסכון ייעודי
    "ira", "pension",
    # מונחי הלוואה/גבייה טכניים
    "defer", "prepay",
    # לשכות אשראי (שמות ספציפיים - סימן חזק לאוריינות)
    "transunion", "equifax", "experian",
    # חינוך פיננסי מוכר בשם
    "ramsey", "snowball",
}

UNCERTAIN_WORDS = {
    "ratio": "בהקשר זה כנראה 'יחס חוב-הכנסה', אבל מילה כללית",
    "consolidate": "מונח נכון, אך כה נפוץ ב-LendingClub (קטגורית מטרה מובנית בפלטפורמה) שאין לו כוח הבחנה",
    "consolidation": "כנ״ל",
    "secure": "עלול להיות 'secured debt' אחרי למטיזציה (secured->secure) או סתם 'secure job/feel secure'",
    "liquid": "עלול להתכוון ל'נזיל' פיננסית או למשמעות מילולית",
    "appraisal": "הערכת שווי (טכני) אך גם יכול להיות תהליך פרוצדורלי לא קשור לידע הלווה",
    "appraise": "כנ״ל",
    "variable": "'ריבית משתנה' אפשרי, אך גם שם תואר כללי",
    "fix": "'ריבית קבועה' אפשרי, אך בקורפוס הזה לרוב 'לתקן' (הקשר שיפוצים חזק)",
    "net": "'הכנסה נטו' אפשרי, אך גם מילה כללית (רשת, net כקיצור לאינטרנט)",
    "term": "'תקופת הלוואה' אפשרי, אך מילה אנגלית כללית ביותר",
    "strategy": "'אסטרטגיה פיננסית' אפשרי אבל כללי מדי לבד",
    "portfolio": "תיק השקעות (טכני) אבל גם 'תיק עבודות' יצירתי (יש בקורפוס גם photographer/designer)",
    "retirement": "מושג פרישה - יכול לבטא תכנון פיננסי או סתם ציון שלב חיים",
    "withdrawal": "משיכת כספים (טכני) אך גם משיכה כללית",
    "contribution": "'הפקדה לפנסיה' אפשרי אבל כללי מדי",
    "deduction": "'ניכוי מס' אפשרי אבל גם ניכוי לוגי/שכר כללי",
    "worthy": "'credit-worthy' אפשרי אבל מילה כללית מאוד ('worthy cause' וכו')",
    "inquiry": "'בירור אשראי' (hard/soft inquiry) אפשרי אבל גם שאלה כללית",
    "bureau": "'לשכת אשראי' אפשרי אבל גם רהיט (bureau) או משרד ממשלתי",
    "equity": "'הון עצמי בבית' לרוב, אבל יכול גם להתכוון ל'הוגנות' באנגלית כללית",
}

freq = (
    pd.read_csv(DICT_DIR / "vocabulary_frequencies_lending_club.csv")
    .query("Type == 'Unigram'")
    .set_index("Term")["Document_Frequency"]
)

full = pd.read_csv(DICT_DIR / "vocabulary_frequencies_lending_club.csv")
full["Financial_Literacy"] = "Regular"
is_unigram = full["Type"] == "Unigram"
full.loc[is_unigram & full["Term"].isin(LITERACY_WORDS), "Financial_Literacy"] = "Literacy"
full.loc[is_unigram & full["Term"].isin(UNCERTAIN_WORDS.keys()), "Financial_Literacy"] = "Uncertain"

full["Reason"] = full["Term"].map(UNCERTAIN_WORDS).fillna("")

full.to_csv(DICT_DIR / "LC_Financial_Literacy_Dictionary.csv", index=False, encoding="utf-8-sig")

print(f"Literacy:  {(full['Financial_Literacy'] == 'Literacy').sum()}")
print(f"Uncertain: {(full['Financial_Literacy'] == 'Uncertain').sum()}")
print("Output file: LC_Financial_Literacy_Dictionary.csv")
