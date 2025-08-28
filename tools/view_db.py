import sqlite3

if __name__ == '__main__':
    sqliteConnection = sqlite3.connect('captcha_images.db')
    sql_query = """SELECT name FROM sqlite_master  
    WHERE type='table';"""
    cursor = sqliteConnection.cursor()
    cursor.execute(sql_query)
    print(cursor.fetchall())