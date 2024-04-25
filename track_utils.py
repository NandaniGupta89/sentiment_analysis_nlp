# Load Database Pkg
import sqlite3

# Function to create the pageTrackTable
def create_page_visited_table():
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS pageTrackTable(pagename TEXT,timeOfvisit TIMESTAMP)')
    conn.commit()
    conn.close()

# Function to add page visited details
def add_page_visited_details(pagename, timeOfvisit):
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute('INSERT INTO pageTrackTable(pagename, timeOfvisit) VALUES (?, ?)', (pagename, timeOfvisit))
    conn.commit()
    conn.close()

# Function to view all page visited details
def view_all_page_visited_details():
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute('SELECT * FROM pageTrackTable')
    data = c.fetchall()
    conn.close()
    return data

# Function to create the emotionclfTable
def create_emotionclf_table():
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS emotionclfTable(rawtext TEXT, prediction TEXT, probability NUMBER, timeOfvisit TIMESTAMP)')
    conn.commit()
    conn.close()

# Function to add prediction details
def add_prediction_details(rawtext, prediction, probability, timeOfvisit):
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute('INSERT INTO emotionclfTable(rawtext, prediction, probability, timeOfvisit) VALUES (?, ?, ?, ?)', (rawtext, prediction, probability, timeOfvisit))
    conn.commit()
    conn.close()

# Function to view all prediction details
def view_all_prediction_details():
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute('SELECT * FROM emotionclfTable')
    data = c.fetchall()
    conn.close()
    return data

def main():
    create_page_visited_table()

if __name__ == "__main__":
    main()
