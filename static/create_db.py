import sqlite3

# Connect to SQLite database
conn = sqlite3.connect("disease_data.db")
cursor = conn.cursor()

# Create a table for diseases
cursor.execute('''
CREATE TABLE IF NOT EXISTS diseases (
    name TEXT PRIMARY KEY,
    precautions TEXT,
    state TEXT,
    fatality_score INTEGER,
    treatment TEXT
)
''')

# Insert sample data (you can extend this)
cursor.executemany('''
INSERT OR IGNORE INTO diseases (name, precautions, state, fatality_score, treatment) 
VALUES (?, ?, ?, ?, ?)
''', [
    ("Flu", "Drink plenty of fluids,Get rest,Take antiviral medication", "Moderate", 2, "Supportive care, antiviral drugs if needed"),
    ("COVID-19", "Wear a mask,Isolate if positive,Get vaccinated", "Severe", 7, "Oxygen therapy, antiviral drugs, ICU if needed")
])

conn.commit()
conn.close()

print("Database and sample data created successfully!")
